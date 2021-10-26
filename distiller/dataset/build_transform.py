from torchvision import transforms

class ApplyTransform:
    MEAN = {'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
        'cinic10': (0.47889522, 0.47227842, 0.43047404),
        'stl10': (0.4914, 0.4822, 0.4465),
        'svhn': (0.4377, 0.4438, 0.4728),
        'tinyimagenet': (0.485, 0.456, 0.406),
        'imagenet': (0.485, 0.456, 0.406)}

    STD = {'cifar10': (0.2470, 0.2435, 0.2616),
       'cifar100': (0.2675, 0.2565, 0.2761),
       'cinic10': (0.24205776, 0.23828046, 0.25874835),
       'stl10': (0.2471, 0.2435, 0.2616),
       'svhn': (0.1980, 0.2010, 0.1970),
       'tinyimagenet': (0.229, 0.224, 0.225),
       'imagenet': (0.229, 0.224, 0.225)}
    
    def __init__(self, opt, split):
        self.mean = self.MEAN[opt.dataset]
        self.std = self.STD[opt.dataset]
        
        self.transform = self.standard_transform(opt, split)
        if hasattr(opt, 'distill'):
            if opt.distill in ['ifacrd', 'ifacrdv2'] and opt.simclr_aug and split == 'train':
                self.transform_agg = self.simclr_transform(opt)
            
    def __call__(self, x):
        if hasattr(self, 'transform_agg'):
            return self.transform(x), self.transform_agg(x), self.transform_agg(x)
        else:
            return self.transform(x)

    def standard_transform(self, opt, split):
        t = []
        if split == 'train':
            if opt.dataset in ['imagenet']:
                t.append(transforms.RandomResizedCrop(opt.image_size))
            elif opt.image_size == 32 and opt.dataset in ['cifar10', 'cifar100', 'cinic10', 'svhn']:
                t.append(transforms.RandomCrop(opt.image_size, padding=4))
            else:
                t.append(transforms.Resize(opt.image_size+32))
                t.append(transforms.RandomCrop(opt.image_size))
            t.append(transforms.RandomHorizontalFlip())
        else:
            if opt.dataset in ['imagenet']:
                t.append(transforms.Resize(opt.image_size+32))
                t.append(transforms.CenterCrop(opt.image_size))
            elif opt.image_size != 32 or opt.dataset in ['stl10', 'tinyimagenet']:
                t.append(transforms.Resize(opt.image_size))
                            
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(t)

    def simclr_transform(self, opt):
        p_blur = 0.5 if opt.image_size > 32 else 0 # exclude cifar
        s = 1 # 0.5 for simsiam, 1 for simclr
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        t = []
        if opt.dataset in ['imagenet']:
            t.append(transforms.RandomResizedCrop(opt.image_size))
        elif opt.image_size == 32 and opt.dataset in ['cifar10', 'cifar100', 'cinic10', 'svhn']:
            t.append(transforms.RandomCrop(opt.image_size, padding=4))
        else:
            t.append(transforms.Resize(opt.image_size+32))
            t.append(transforms.RandomCrop(opt.image_size))

        transform = [
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=opt.image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
                # We blur the image 50% of the time using a Gaussian kernel. We randomly sample σ ∈ [0.1, 2.0], and the kernel size is set to be 10% of the image height/width.
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]

        [t.append(tr) for tr in transform]      
            
        return transforms.Compose(t)
