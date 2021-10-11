from torch.utils.data import DataLoader

from DataPreparation import DataPreparation
from TransUNetModel import VisionTransformer

if __name__ == '__main__':
    print("program started")
    DATA_DIRECTORY = 'Datasets/'
    DATASETS = ['B5', 'B39', 'EM', 'ssTEM', 'TNBC']
    TARGET = ['TNBC']
    BATCH_SIZE = 16

    dataPreparation = DataPreparation(DATA_DIRECTORY, DATASETS, TARGET)

    print("load data")
    images_train, groundtruth_train, images_valid, groundtruth_valid = dataPreparation.load_data()

    train_loader = DataLoader(images_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(groundtruth_train, batch_size=BATCH_SIZE, shuffle=True)

    print("load model")
    attention_dropout_rate = 0.0
    classifier = "seg"
    decoder_channels = (256, 128, 64, 16)
    dropout_rate = 0.1
    grid_size = (16, 16)
    hidden_size = 1024
    mlp_dim = 4096
    n_classes = 2
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

    print("start training")

    print("training ended")
