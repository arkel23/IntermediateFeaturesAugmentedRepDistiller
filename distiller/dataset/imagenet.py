from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity
from PIL import Image

"""
# https://github.com/winycg/HSAKD/blob/main/eval_rep.py
mean = {
    'imagenet': (0.485, 0.456, 0.406),
}
std = {
    'imagenet': (0.229, 0.224, 0.225),
}
"""

class ImageNet(datasets.ImageFolder):
    """`ImageNet <https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
            
    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    sample_image = os.path.join('val', 'n01440764', 'ILSVRC2012_val_00000293.JPEG')
    
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform= None,
    ):
        integrity = self._check_integrity(root)
        if not integrity:
            raise RuntimeError('File not found or corrupted.')
            
        if train:
            dataset_path = os.path.join(root, 'train')
        else:
            dataset_path = os.path.join(root, 'val')
        
        super(ImageNet, self).__init__(root=dataset_path, transform=transform,
                                      target_transform=target_transform)
        
        self.parent_root = root
        self.train = train  # training set or test set

    def _check_integrity(self, root):
        check_path = os.path.join(root, self.sample_image)
        if check_integrity(check_path):
            print('Files already downloaded and verified')
            return True
        return False
    
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class ImageNetInstance(ImageNet):
    """ImageNetInstance Dataset.
    """
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class ImageNetInstanceSample(ImageNet):
    """
    ImageNetInstance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 200
        num_samples = len(self.samples)
        label = self.targets
        
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
        img_path, target = self.samples[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)
        img = Image.open(img_path).convert('RGB')
        
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