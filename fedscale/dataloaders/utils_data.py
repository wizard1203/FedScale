# -*- coding: utf-8 -*-

import sys

import torch
import numpy as np
from torchvision import transforms


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def get_data_transform(data: str):
    if data == 'mnist':
        train_transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif data == 'cifar':
        train_transform = transforms.Compose([
            # input arguments: length&width of a figure
            transforms.RandomCrop(32, padding=4),
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # convert PIL image or numpy.ndarray to tensor
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif data == 'ILSVRC2012_hdf5':
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        image_size = 224
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        train_transform.transforms.append(Cutout(16))
        test_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    elif data == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),   # input arguments: length&width of a figure
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,  # convert PIL image or numpy.ndarray to tensor
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize,
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif data == 'emnist':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            # transforms.Resize(224),   # input arguments: length&width of a figure
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),  # convert PIL image or numpy.ndarray to tensor
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            # transforms.Resize(224),
            # transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif data == 'openImg':
        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.RandomResizedCrop((128,128)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    elif data == 'openImgInception':
        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(64),
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Resize(224),   # input arguments: length&width of a figure
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),  # convert PIL image or numpy.ndarray to tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Resize(224),
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif data == 'inaturalist':
        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(64),
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Resize(224),   # input arguments: length&width of a figure
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),  # convert PIL image or numpy.ndarray to tensor
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Resize(224),
            # transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        print('Data must be {} or {} !'.format('mnist', 'cifar'))
        sys.exit(-1)

    return train_transform, test_transform
