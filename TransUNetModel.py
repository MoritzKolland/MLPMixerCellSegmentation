import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.modules.utils import _pair

from vit_seg_modeling_resnet_skip import ResNetV2


class Attention(nn.Module):
    def __init__(self, vis, num_attention_head, hidden_size, attention_dropout_rate):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_attention_head
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = functional.gelu
        self.dropout = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, img_size, grid_size, hidden_size, dropout_rate, num_resnet_layers, width_factor):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)

        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])

        self.hybrid_model = ResNetV2(block_units=num_resnet_layers, width_factor=width_factor)
        in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.patch_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings, features


class Block(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate, vis, num_attention_head, attention_dropout_rate):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size=hidden_size, mlp_dim=mlp_dim, dropout_rate=dropout_rate)
        self.attn = Attention(vis=vis, num_attention_head=num_attention_head, hidden_size=hidden_size,
                              attention_dropout_rate=attention_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        return x, weights


class Encoder(nn.Module):
    def __init__(self, vis, hidden_size, num_transformer_layers, mlp_dim,
                 dropout_rate, num_attention_head, attention_dropout_rate):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_transformer_layers):
            layer = Block(hidden_size=hidden_size, mlp_dim=mlp_dim,
                          dropout_rate=dropout_rate, vis=vis,
                          num_attention_head=num_attention_head,
                          attention_dropout_rate=attention_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encode = self.encoder_norm(hidden_states)

        return encode, attn_weights


class Transformer(nn.Module):
    def __init__(self, img_size, grid_size, hidden_size, dropout_rate, vis,
                 num_transformer_layers, num_resnet_layers, mlp_dim,
                 num_attention_head, attention_dropout_rate, width_factor):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size, grid_size=grid_size,
                                     hidden_size=hidden_size, dropout_rate=dropout_rate,
                                     num_resnet_layers=num_resnet_layers, width_factor=width_factor)
        self.encoder = Encoder(vis=vis, hidden_size=hidden_size, num_transformer_layers=num_transformer_layers,
                               mlp_dim=mlp_dim, dropout_rate=dropout_rate,
                               num_attention_head=num_attention_head,
                               attention_dropout_rate=attention_dropout_rate)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encode, attn_weights = self.encoder(embedding_output)

        return encode, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels=in_channels + skip_channels, out_channels=out_channels,
                                kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=2):
        conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, hidden_size, decoder_channels, n_skip, skip_channels):
        super().__init__()
        head_channels = 512
        self.conf_more = Conv2dReLU(hidden_size, head_channels, kernel_size=3, padding=1, use_batchnorm=True)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        self.n_skip = n_skip
        if n_skip != 0:
            for i in range(4 - n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch
                  in zip(in_channels, out_channels, skip_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        b, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch), int(np.sqrt(n_patch)))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(b, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, classifier, grid_size, hidden_size, dropout_rate,
                 num_transformer_layers, num_resnet_layer, mlp_dim, num_attention_head,
                 attention_dropout_rate, decoder_channels, n_skip,
                 skip_channels, n_classes, width_factor, img_size=256,
                 num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = classifier
        self.transformer = Transformer(img_size=img_size, grid_size=grid_size,
                                       hidden_size=hidden_size, dropout_rate=dropout_rate,
                                       vis=vis, num_transformer_layers=num_transformer_layers,
                                       num_resnet_layers=num_resnet_layer, mlp_dim=mlp_dim,
                                       num_attention_head=num_attention_head,
                                       attention_dropout_rate=attention_dropout_rate,
                                       width_factor=width_factor)
        self.decoder = DecoderCup(hidden_size=hidden_size, decoder_channels=decoder_channels,
                                  n_skip=n_skip, skip_channels=skip_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=n_classes, kernel_size=3)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features, = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)

        return logits
