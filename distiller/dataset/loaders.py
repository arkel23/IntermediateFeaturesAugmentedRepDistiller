import os

from torch.utils import data
from torchvision import datasets

from .build_transform import ApplyTransform
from .cifar10 import CIFAR10Instance, CIFAR10InstanceSample
from .cifar100 import CIFAR100Instance, CIFAR100InstanceSample
from .stl10 import STL10Instance, STL10InstanceSample
from .svhn import SVHNInstance, SVHNInstanceSample
from .cinic10 import CINIC10, CINIC10Instance, CINIC10InstanceSample
from .tinyimagenet import TinyImageNet, TinyImageNetInstance, TinyImageNetInstanceSample
from .imagenet import ImageNet, ImageNetInstance, ImageNetInstanceSample

def build_dataloaders(opt, vanilla=True):
    
    os.makedirs(opt.dataset_path, exist_ok=True)
    
    train_transform = ApplyTransform(split='train', opt=opt)
    val_transform = ApplyTransform(split='val', opt=opt)
    
    train_set = get_train_set(opt.dataset, opt.dataset_path, train_transform, opt, vanilla)
    val_set, n_cls = get_val_set(opt.dataset, opt.dataset_path, val_transform)
    n_data = len(train_set)
    
    if opt.distributed:
        train_sampler = data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)    
    val_loader = data.DataLoader(val_set, batch_size=64, shuffle=False, 
        num_workers=int(opt.num_workers/2), pin_memory=True)
    
    if vanilla:
        return train_loader, val_loader, n_cls
    return train_loader, val_loader, n_cls, n_data


def get_val_set(dataset, dataset_path, transform):
    
    if dataset == 'cifar10':
        val_set = datasets.CIFAR10(root=dataset_path, download=True,
            train=False, transform=transform)
        n_cls = 10
    elif dataset == 'cifar100':
        val_set = datasets.CIFAR100(root=dataset_path, download=True,
            train=False, transform=transform)
        n_cls = 100
    elif dataset == 'stl10':
        val_set = datasets.STL10(root=dataset_path, download=True,
            split='test', transform=transform)
        n_cls = 10
    elif dataset == 'svhn':
        val_set = datasets.SVHN(root=dataset_path, download=True,
            split='test', transform=transform)
        n_cls = 10
    elif dataset == 'cinic10':
        val_set = CINIC10(root=dataset_path, download=True,
            train=False, transform=transform)
        n_cls = 10
    elif dataset == 'tinyimagenet':
        val_set = TinyImageNet(root=dataset_path, download=True,
            train=False, transform=transform)
        n_cls = 200
    elif dataset == 'imagenet':
        val_set = ImageNet(root=dataset_path,
            train=False, transform=transform)
        n_cls = 1000
    else:
        raise NotImplementedError
    
    return val_set, n_cls


def get_train_set(dataset, dataset_path, transform, opt, vanilla):
    
    if vanilla:
        if dataset == 'cifar10':
            train_set = datasets.CIFAR10(root=dataset_path, download=True,
                train=True, transform=transform)
        elif dataset == 'cifar100':
            train_set = datasets.CIFAR100(root=dataset_path, download=True,
                train=True, transform=transform)
        elif dataset == 'stl10':
            train_set = datasets.STL10(root=dataset_path, download=True,
                split='test', transform=transform)
        elif dataset == 'svhn':
            train_set = datasets.SVHN(root=dataset_path, download=True,
                split='test', transform=transform)
        elif dataset == 'cinic10':
            train_set = CINIC10(root=dataset_path, download=True,
                train=True, transform=transform)
        elif dataset == 'tinyimagenet':
            train_set = TinyImageNet(root=dataset_path, download=True,
                train=True, transform=transform)
        elif dataset == 'imagenet':
            train_set = ImageNet(root=dataset_path,
                train=True, transform=transform)
        else:
            raise NotImplementedError
    else:
        if dataset == 'cifar10':
            if opt.distill in ['crd']:
                train_set = CIFAR10InstanceSample(
                    root=dataset_path, download=True, train=True, transform=transform,
                    k=opt.nce_k, mode=opt.mode, is_sample=True, percent=1.0)
            
            else:
                train_set = CIFAR10Instance(root=dataset_path, download=True,
                    train=True, transform=transform)
        elif dataset == 'cifar100':
            if opt.distill in ['crd']:
                train_set = CIFAR100InstanceSample(
                    root=dataset_path, download=True, train=True, transform=transform,
                    k=opt.nce_k, mode=opt.mode, is_sample=True, percent=1.0)
            else:
                train_set = CIFAR100Instance(root=dataset_path, download=True,
                    train=True, transform=transform)
        elif dataset == 'stl10':
            if opt.distill in ['crd']:
                train_set = STL10InstanceSample(
                    root=dataset_path, download=True, split='train', transform=transform,
                    k=opt.nce_k, mode=opt.mode, is_sample=True, percent=1.0)
            else:
                train_set = STL10Instance(root=dataset_path, download=True,
                    split='train', transform=transform)
        elif dataset == 'svhn':
            if opt.distill in ['crd']:
                train_set = SVHNInstanceSample(
                    root=dataset_path, download=True, split='train', transform=transform,
                    k=opt.nce_k, mode=opt.mode, is_sample=True, percent=1.0)
            else:
                train_set = SVHNInstance(root=dataset_path, download=True,
                    split='train', transform=transform)
        elif dataset == 'cinic10':
            if opt.distill in ['crd']:
                train_set = CINIC10InstanceSample(
                    root=dataset_path, download=True, train=True, transform=transform,
                    k=opt.nce_k, mode=opt.mode, is_sample=True, percent=1.0)
            else:
                train_set = CINIC10Instance(root=dataset_path, download=True,
                    train=True, transform=transform)
        elif dataset == 'tinyimagenet':
            if opt.distill in ['crd']:
                train_set = TinyImageNetInstanceSample(
                    root=dataset_path, download=True, train=True, transform=transform,
                    k=opt.nce_k, mode=opt.mode, is_sample=True, percent=1.0)
            else:
                train_set = TinyImageNetInstance(root=dataset_path, download=True,
                    train=True, transform=transform)
        elif dataset == 'imagenet':
            if opt.distill in ['crd']:
                train_set = ImageNetInstanceSample(
                    root=dataset_path, download=True, train=True, transform=transform,
                    k=opt.nce_k, mode=opt.mode, is_sample=True, percent=1.0)
            else:
                train_set = ImageNetInstance(root=dataset_path, download=True,
                    train=True, transform=transform)
        else:
            raise NotImplementedError
        
    return train_set
