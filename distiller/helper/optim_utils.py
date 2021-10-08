import math

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

class Scheduler():
    def __init__(self, opt, optimizer):
        self.opt = opt
        self.optimizer = optimizer

    def step(self, epoch):
        new_lr = self.get_epoch_values(epoch)
        self.adjust_learning_rate(new_lr)
        
    def get_epoch_values(self, epoch):
        stage = math.ceil((epoch - self.opt.warmup_epochs) / self.opt.decay_epochs)
        if stage < 1:
            return self.opt.lr
        else:
            return self.opt.lr * (self.opt.decay_rate ** stage)
        
    def adjust_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def return_optimizer_scheduler(opt, model):

    if opt.sched == 'step':
        opt.warmup_lr = opt.lr
    elif opt.sched == 'cosine':
        opt.warmup_lr = 1e-6
    
    opt.opt_eps = 1e-8
    opt.opt_betas = None
    
    opt.lr_noise = None
    opt.lr_noise_pct = 0.67
    opt.lr_noise_std = 1.0
    opt.min_lr = 1e-5
    opt.cooldown_epochs = 10
    opt.patience_epochs = 10
    
    optimizer = create_optimizer(opt, model)
    
    if opt.sched == 'warmup_step':
        lr_scheduler = Scheduler(opt, optimizer)
    else:
        lr_scheduler, _ = create_scheduler(opt, optimizer)
    
    return optimizer, lr_scheduler