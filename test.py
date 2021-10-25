import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from DataPreparation import DataPreparation
from TransUNetModel import VisionTransformer

print("program started")
DATA_DIRECTORY = 'Datasets/'
DATASETS = ['B5', 'B39', 'EM', 'ssTEM', 'TNBC']
TARGET = ['TNBC']
BATCH_SIZE = 16

print("load data")
train_data = DataPreparation(DATA_DIRECTORY, DATASETS, TARGET, 'train')
val_data = DataPreparation(DATA_DIRECTORY, DATASETS, TARGET, 'valid')

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

print("load model")
attention_dropout_rate = 0.0
classifier = "seg"
decoder_channels = (256, 128, 64, 16)
dropout_rate = 0.1
grid_size = (16, 16)
hidden_size = 1024
mlp_dim = 4096
n_classes = 1
n_skip = 3
num_attention_head = 16
num_resnet_layers = (3, 4, 9)
num_transformer_layer = 24
skip_channels = [512, 256, 64, 16]
width_factor = 1

model = VisionTransformer(attention_dropout_rate=attention_dropout_rate,
                          classifier=classifier,
                          decoder_channels=decoder_channels,
                          dropout_rate=dropout_rate,
                          grid_size=grid_size,
                          hidden_size=hidden_size,
                          mlp_dim=mlp_dim,
                          n_classes=n_classes,
                          n_skip=n_skip,
                          num_attention_head=num_attention_head,
                          num_resnet_layer=num_resnet_layers,
                          num_transformer_layers=num_transformer_layer,
                          skip_channels=skip_channels,
                          width_factor=width_factor
                          )

model.load_state_dict(torch.load("state_dict.pt"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.eval()

for i in range(10):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)

        fig = plt.figure()
        fig.add_subplot(3, 1, 1)
        img = images.numpy()[0][0]
        plt.imshow(img)
        fig.add_subplot(3, 1, 2)
        val = labels.numpy()[0][0]
        plt.imshow(val)
        fig.add_subplot(3, 1, 3)
        out = output.detach().numpy()[0][0]
        plt.imshow(out)

        input("Press Enter to continue...")
