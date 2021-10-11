from DataPreparation import DataPreparation
from torch.utils.data import DataLoader

if __name__ == '__main__':
    print("program started")
    directory = 'Datasets/'
    datasets = ['B5', 'B39', 'EM', 'ssTEM', 'TNBC']
    target = 'TNBC'
    BATCH_SIZE = 128

    dataPreparation = DataPreparation(directory, datasets, target)

    print("load data")
    images_train, groundtruth_train, images_valid, groundtruth_valid = dataPreparation.load_data()

    train_loader = DataLoader(images_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(groundtruth_train, batch_size=BATCH_SIZE, shuffle=False)

    print("load model")

    print("start training")

    print("training ended")
