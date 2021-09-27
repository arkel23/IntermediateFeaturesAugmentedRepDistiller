from .cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

def build_dataloader(opt, vanilla=True):
    if vanilla:
        if opt.dataset == 'cifar10':
            train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 10
        elif opt.dataset == 'cifar100':
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 100
        else:
            raise NotImplementedError(opt.dataset)
        
        return train_loader, val_loader, n_cls
    
    else:
        if opt.dataset == 'cifar10':
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data = get_cifar10_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                train_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 10
        elif opt.dataset == 'cifar100':
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 100
        else:
            raise NotImplementedError(opt.dataset)
        
        return train_loader, val_loader, n_cls, n_data