import os

from .cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from .cinic10 import get_cinic10_dataloaders, get_cinic10_dataloaders_sample
from .stl10 import get_stl10_dataloaders, get_stl10_dataloaders_sample
from .svhn import get_svhn_dataloaders, get_svhn_dataloaders_sample
from .tinyimagenet import get_tinyimagenet_dataloaders, get_tinyimagenet_dataloaders_sample
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample

def build_dataloader(opt, vanilla=True):
    
    if opt.dataset_path:
        dataset_path = opt.dataset_path
    else:
        dataset_path = './data/'
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    
    if vanilla:
        if opt.dataset == 'cifar10':
            train_loader, val_loader = get_cifar10_dataloaders(dataset_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 10
        elif opt.dataset == 'cifar100':
            train_loader, val_loader = get_cifar100_dataloaders(dataset_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 100
        elif opt.dataset == 'svhn':
            train_loader, val_loader = get_svhn_dataloaders(dataset_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 10
        elif opt.dataset == 'cinic10':
            train_loader, val_loader = get_cinic10_dataloaders(dataset_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 10
        elif opt.dataset == 'stl10':
            train_loader, val_loader = get_stl10_dataloaders(dataset_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 10
        elif opt.dataset == 'tinyimagenet':
            train_loader, val_loader = get_tinyimagenet_dataloaders(dataset_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 200
        elif opt.dataset == 'imagenet':
            train_loader, val_loader = get_imagenet_dataloaders(dataset_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 1000
        else:
            raise NotImplementedError(opt.dataset)
        
        return train_loader, val_loader, n_cls
    
    else:
        if opt.dataset == 'cifar10':
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data = get_cifar10_dataloaders_sample(dataset_path, batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                train_loader, val_loader, n_data = get_cifar10_dataloaders(dataset_path, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 10
        elif opt.dataset == 'cifar100':
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(dataset_path, batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                train_loader, val_loader, n_data = get_cifar100_dataloaders(dataset_path, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 100
        elif opt.dataset == 'svhn':
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data = get_svhn_dataloaders_sample(dataset_path, batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                train_loader, val_loader, n_data = get_svhn_dataloaders(dataset_path, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 10
        elif opt.dataset == 'cinic10':
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data = get_cinic10_dataloaders_sample(dataset_path, batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                train_loader, val_loader, n_data = get_cinic10_dataloaders(dataset_path, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 10
        elif opt.dataset == 'stl10':
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data = get_stl10_dataloaders_sample(dataset_path, batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                train_loader, val_loader, n_data = get_stl10_dataloaders(dataset_path, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 10
        elif opt.dataset == 'tinyimagenet':
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data = get_tinyimagenet_dataloaders_sample(dataset_path, batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                train_loader, val_loader, n_data = get_tinyimagenet_dataloaders(dataset_path, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 200
        elif opt.dataset == 'imagenet':
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data = get_imagenet_dataloaders_sample(dataset_path, batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                train_loader, val_loader, n_data = get_imagenet_dataloaders(dataset_path, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 1000
        else:
            raise NotImplementedError(opt.dataset)
        
        return train_loader, val_loader, n_cls, n_data