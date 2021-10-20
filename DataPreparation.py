import os

import torch
from PIL import Image
from torchvision import transforms


class DataPreparation:
    def __init__(self, directory, datasets, target, train_valid):
        self.images_train = []
        self.groundtruth_train = []
        self.groundtruth_valid = []
        self.images_valid = []
        self.directory = directory
        self.datasets = datasets
        self.target = target
        self.train_valid = train_valid

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

    def __len__(self):
        if self.train_valid == 'train':
            return len(self.groundtruth_train)
        elif self.train_valid == 'valid':
            return len(self.groundtruth_valid)

    def __getitem__(self, item):
        image = None
        ground_truth = None

        if torch.is_tensor(item):
            item = item.tolist()

        if self.train_valid == 'train':
            image = Image.open(self.images_train[item])
            ground_truth = Image.open(self.groundtruth_train[item])
        elif self.train_valid == 'valid':
            image = Image.open(self.images_valid[item])
            ground_truth = Image.open(self.groundtruth_valid[item])

        image = transforms.ToTensor()(image)
        ground_truth = transforms.ToTensor()(ground_truth)

        return image, ground_truth
