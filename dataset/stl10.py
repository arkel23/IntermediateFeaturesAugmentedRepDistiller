from __future__ import print_function

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

"""
# https://github.com/YU1ut/MixMatch-pytorch/pull/25/files
mean = {
    'STL1010': (0.4914, 0.4822, 0.4465),
}

std = {
    'stl10': (0.2471, 0.2435, 0.2616),
}
"""

class STL10Instance(datasets.STL10):
    """STL10Instance Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_stl10_dataloaders(dataset_path, batch_size=128, num_workers=8, is_instance=False):
    """
    stl 10
    """
    data_folder = dataset_path

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    if is_instance:
        train_set = STL10Instance(root=data_folder,
                                     download=True,
                                     split='train',
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.STL10(root=data_folder,
                                      download=True,
                                      split='train',
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_set = datasets.STL10(root=data_folder,
                                 download=True,
                                 split='test',
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=64,
                             shuffle=False,
                             num_workers=int(num_workers/2),
                             pin_memory=True)

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


class STL10InstanceSample(datasets.STL10):
    """
    STL10Instance+Sample Dataset
    """
    def __init__(self, root, split='train',
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, split=split, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 10
        num_samples = len(self.data)
        label = self.labels
        
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_stl10_dataloaders_sample(dataset_path, batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    stl 10
    """
    data_folder = dataset_path

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    train_set = STL10InstanceSample(root=data_folder,
                                       download=True,
                                       split='train',
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_set = datasets.STL10(root=data_folder,
                                 download=True,
                                 split='test',
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=64,
                             shuffle=False,
                             num_workers=int(num_workers/2),
                             pin_memory=True)

    return train_loader, test_loader, n_data
