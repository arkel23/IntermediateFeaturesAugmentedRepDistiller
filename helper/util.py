from __future__ import print_function

import os
import torch
import wandb
import numpy as np
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

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
    lr_scheduler, _ = create_scheduler(opt, optimizer)
    
    return optimizer, lr_scheduler

def summary_stats(epochs, time_total, best_acc, best_epoch, max_memory, no_params):    
    time_avg = time_total / epochs
    best_time = time_avg * best_epoch
    no_params = no_params / (1e6)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('''Total run time (s): {}
          Average time per epoch (s): {}
          Best accuracy (%): {} at epoch {}. Time to reach this accuracy (s): {}
          Max VRAM consumption (GB): {}
          Total number of parameters in all modules (M): {}
          '''.format(time_total, time_avg, best_acc, best_epoch, 
                     best_time, max_memory, no_params))
    
    wandb.run.summary['time_total'] = time_total
    wandb.run.summary['time_avg'] = time_avg
    wandb.run.summary['best_acc'] = best_acc
    wandb.run.summary['best_epoch'] = best_epoch
    wandb.run.summary['best_time'] = best_time
    wandb.run.summary['max_memory'] = max_memory
    wandb.run.summary['no_params'] = no_params

    wandb.finish()
    

def save_model(opt, model, epoch, acc, mode, optimizer=False, vanilla=True):
    if optimizer:
        state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': acc,
                'optimizer': optimizer.state_dict(),
            }
    else:
        state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': acc,
            }        
        
    if mode == 'best':
        if vanilla:
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
        else:
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
        print('Saving the best model!')
        torch.save(state, save_file)
    elif mode == 'epoch':
        save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        print('==> Saving each {} epochs...'.format(opt.save_freq))
        torch.save(state, save_file)
    elif mode == 'last':
        if vanilla:
            save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
        else:
            save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
        print('Saving last epoch')
        torch.save(state, save_file)  
       
            
def count_params_module_list(module_list):
    return sum([count_params_single(model) for model in module_list])

def count_params_single(model):
    return sum([p.numel() for p in model.parameters()])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    pass
