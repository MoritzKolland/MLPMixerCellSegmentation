import os


class DataPreparation:
    def __init__(self, directory, datasets, target):
        self.images_train = []
        self.groundtruth_train = []
        self.groundtruth_valid = []
        self.images_valid = []
        self.directory = directory
        self.datasets = datasets
        self.target = target

    def load_data(self):
        for dataset in self.datasets:
            groundtruth_prefix = dataset + '/Groundtruth/'
            image_prefix = dataset + '/Image/'

            filenames_groundtruth = sorted(
                [self.directory + groundtruth_prefix + f for f in
                 os.listdir(self.directory + groundtruth_prefix) if f[0] != '.'])
            filenames_images = sorted(
                [self.directory + image_prefix + f for f in
                 os.listdir(self.directory + image_prefix) if f[0] != '.'])[:len(filenames_groundtruth)]

            if dataset == self.target:
                for i in range(len(filenames_images)):
                    self.images_valid.append(filenames_images[i])
                    self.groundtruth_valid.append(filenames_groundtruth[i])
            else:
                for i in range(len(filenames_images)):
                    self.images_train.append(filenames_images[i])
                    self.groundtruth_train.append(filenames_groundtruth[i])

        return self.images_train, self.groundtruth_train, self.images_valid, self.groundtruth_valid
