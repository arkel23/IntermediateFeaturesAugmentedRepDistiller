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
        if hasattr(opt, 'simclr_aug') and self.split == 'train':
            self.mode = 'simclr_train'
        else:
            self.mode = 'default'

        self.mean = self.MEAN[opt.dataset]
        self.std = self.STD[opt.dataset]
        
        self.transform = self.build_transform(opt, split)
            
    def __call__(self, x):
        if self.mode == 'simclr_train':
            return self.transform(x), self.transform(x)
        else:
            return self.transform(x)

    def build_transform(self, opt, split):
        if self.mode == 'simclr_train':
            transform = self.simclr_transform(opt, split)
        else:
            transform = self.standard_transform(opt, split)
        return transform

    def standard_transform(self, opt, split):
        if split == 'train':
            if opt.dataset in ['cifar10', 'cifar100', 'cinic10', 'svhn']:
                transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean, self.std)
                            ])
            else:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(opt.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
        else:
            if opt.dataset in ['cifar10', 'cifar100', 'cinic10', 'svhn']:                
                transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean, self.std)
                            ])
            elif opt.dataset in ['stl10', 'tinyimagenet']:
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(self. mean, self.std)
                ])
            else:
                transform = transforms.Compose([
                                transforms.Resize(opt.image_size+32),
                                transforms.CenterCrop(opt.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean, self.std),
                            ])
        return transform

    def simclr_transform(self, opt):
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        transform = transforms.Compose([
                transforms.RandomResizedCrop(size=opt.image_size),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                # We blur the image 50% of the time using a Gaussian kernel. We randomly sample σ ∈ [0.1, 2.0], and the kernel size is set to be 10% of the image height/width.
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean,
                                    std=self.std)
            ])
            
        return transform
