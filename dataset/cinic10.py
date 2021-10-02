from __future__ import print_function

import os
import glob
import numpy as np
from shutil import copyfile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive
from PIL import Image

"""
mean = {
    'cinic10': (0.47889522, 0.47227842, 0.43047404),
}

std = {
    'cinic10': (0.24205776, 0.23828046, 0.25874835),
}
"""

class CINIC10(datasets.ImageFolder):
    """`CINIC10 <https://github.com/BayesWatch/cinic-10>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cinic-10-trainalrge`` exists or will be saved to if download is set to True.
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
    url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    filename = "CINIC-10.tar.gz"
    extended_dir = 'train-extended'
    sample_image = 'cifar10-train-10008.png'
    
    def __init__(
            self,
            root: str,
            train: bool = True,
            download: bool = True,
            transform = None,
            target_transform= None,
    ):
        if download:
            self.download(root)
        integrity = self._check_integrity(root)
        if not integrity:
            raise RuntimeError('File not found or corrupted.')
            
        if train:
            dataset_path = os.path.join(root, self.extended_dir)
        else:
            dataset_path = os.path.join(root, 'test')
        
        super(CINIC10, self).__init__(root=dataset_path, transform=transform,
                                      target_transform=target_transform)
        
        self.parent_root = root
        self.train = train  # training set or test set

    def _check_integrity(self, root):
        check_path = os.path.join(root, self.extended_dir, 'airplane', self.sample_image)
        if check_integrity(os.path.join(root, self.filename)):
            if check_integrity(check_path):
                return True
            else:
                extract_archive(os.path.join(root, self.filename))
                self.combine_train_val(root)
                if check_integrity(check_path):
                    return True
        return False
    
    def download(self, root):
        if self._check_integrity(root):
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, root, filename=self.filename)
        self.combine_train_val(root)
        
    def combine_train_val(self, root):
        enlarge_directory = os.path.join(root, self.extended_dir)
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        sets = ['train', 'valid']
        os.makedirs(enlarge_directory, exist_ok=True)
            
        for s in sets:
            for c in classes:
                os.makedirs(os.path.join(enlarge_directory, c), exist_ok=True)

                source_directory = os.path.join(root, s, c)
                filenames = glob.glob(os.path.join(source_directory, '*.png'))
                for source_fn in filenames:
                    base_fn = os.path.basename(source_fn)
                    dest_fn = os.path.join(enlarge_directory, c, base_fn)
                    copyfile(source_fn, dest_fn)
      
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
                    

class CINIC10Instance(CINIC10):
    """CINIC10Instance Dataset.
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


def get_cinic10_dataloaders(dataset_path, batch_size=128, num_workers=8, is_instance=False):
    """
    cinic 10
    """
    data_folder = dataset_path

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
    ])

    if is_instance:
        train_set = CINIC10Instance(root=data_folder,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = CINIC10(root=data_folder,
                                      train=True,
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_set = CINIC10(root=data_folder,
                                 train=False,
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


class CINIC10InstanceSample(CINIC10):
    """
    CINIC10Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 10
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


def get_cinic10_dataloaders_sample(dataset_path, batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    cinic 10
    """
    data_folder = dataset_path

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
    ])

    train_set = CINIC10InstanceSample(root=data_folder,
                                       train=True,
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

    test_set = CINIC10(root=data_folder,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=64,
                             shuffle=False,
                             num_workers=int(num_workers/2),
                             pin_memory=True)

    return train_loader, test_loader, n_data
