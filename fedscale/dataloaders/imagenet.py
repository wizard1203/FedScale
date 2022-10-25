from __future__ import print_function

import csv
import os
import os.path
import warnings

import h5py
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

import pickle
from PIL import Image

class ImageNet_hdf5():

    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    # def __init__(self, root, dataset='train', transform=None, target_transform=None, imgview=False):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = root
        self.transform = transform
        #self.mapping = {idx:file for idx, file in enumerate(raw_data)}
        hdf5fn = os.path.join(self.root)
        self.hf = h5py.File(hdf5fn, "r", libver="latest", swmr=True)
        self.t = "train" if train else "val"
        self.n_images = self.hf["%s_img" % self.t].shape[0]
        self.targets = self.hf["%s_labels" % self.t][...]
        self.data = self.hf["%s_img" % self.t]
        self.transform = transform
        self.target_transform = target_transform

    def _get_dataset_x_and_target(self, index):
        img = self.data[index, ...]
        target = self.targets[index]
        return img, np.int64(target)

    def __getitem__(self, index):
        img, target = self._get_dataset_x_and_target(index)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.n_images

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    # # def load_meta_data(self, path):
    # def load_meta_data(self, net_dataidx_map_file):
    #     datas, labels = [], []
    #     # with open(path) as csv_file:
    #     #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     #     line_count = 0
    #     #     for row in csv_reader:
    #     #         if line_count != 0:
    #     #             datas.append(row[1])
    #     #             labels.append(int(row[-1]))
    #     #         line_count += 1
    #     with open(net_dataidx_map_file, 'rb') as f:
    #         net_dataidx_map = pickle.load(f)
    #     return datas, labels

    # # def load_file(self, path):
    # def load_file(self, net_dataidx_map_file):

    #     # load meta file to get labels
    #     datas, labels = self.load_meta_data(os.path.join(net_dataidx_map_file)
    #     return datas, labels
