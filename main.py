from DataPreparation import DataPreparation

if __name__ == '__main__':
    print("program started")
    directory = 'Datasets/'
    datasets = ['B5', 'B39', 'EM', 'ssTEM', 'TNBC']
    target = 'TNBC'
    dataPreparation = DataPreparation(directory, datasets, target)

    print("load data")
    images_train, groundtruth_train, images_valid, groundtruth_valid = dataPreparation.load_data()

    print("load model")

    print("start training")

    print("training ended")
