from __future__ import print_function, division

import copy
import time
import sys
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .misc_utils import AverageMeter
from .dist_utils import reduce_tensor
from .optim_utils import return_optimizer_scheduler

def init(model_s, model_t, init_modules, criterion, train_loader, opt):
    # create a copy of args since lr is changed only in this module
    opt = copy.deepcopy(opt)
    
    model_t.eval()
    model_s.eval()
    init_modules.train()

    model_t.to(opt.device)
    model_s.to(opt.device)
    init_modules.to(opt.device)
    if opt.distributed:
        model_s = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_s)
        model_t = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_t)
        init_modules = torch.nn.SyncBatchNorm.convert_sync_batchnorm(init_modules)
        model_s = DDP(model_s, device_ids=[opt.local_rank])
        model_t = DDP(model_t, device_ids=[opt.local_rank])
        init_modules = DDP(init_modules, device_ids=[opt.local_rank])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if opt.model_s in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
        'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and opt.distill == 'factor':
        opt.base_lr = opt.base_lr / 5
        if opt.distributed:
            opt.lr = opt.base_lr * ((opt.world_size * opt.batch_size) / 256)
        else:
            opt.lr = opt.base_lr * (opt.batch_size / 256)
    optimizer, _ = return_optimizer_scheduler(opt, init_modules)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    for epoch in range(1, opt.init_epochs + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()
        for idx, data in enumerate(train_loader):
            if opt.distill in ['crd']:
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                index = index.cuda()
                if opt.distill in ['crd']:
                    contrast_idx = contrast_idx.cuda()

            # ============= forward ==============
            out_s = model_s(input, classify_only=False)
            feat_s = out_s[:-1]
            
            with torch.no_grad():
                out_t = model_t(input, classify_only=False)
                feat_t = out_t[:-1]
                if opt.distill != 'ifacrd':
                    feat_t = [f.detach() for f in feat_t]
        

            if opt.distill == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss_group = criterion(g_s, g_t)
                loss = sum(loss_group)
            elif opt.distill == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)
                loss = criterion(f_t_rec, f_t)
            elif opt.distill == 'fsp':
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                loss = sum(loss_group)
            else:
                raise NotImplemented('Not supported in init training: {}'.format(opt.distill))

            torch.cuda.synchronize()
            
            if opt.distributed:
                reduced_loss = reduce_tensor(loss.data, opt.world_size)
            else:
                reduced_loss = loss.data
            losses.update(reduced_loss.item(), input.size(0))
            
            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        # end of epoch
        if opt.local_rank == 0:
            wandb.log({'init_train_loss': losses.avg})
            print('Epoch: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
                epoch, opt.init_epochs, batch_time=batch_time, losses=losses))
            sys.stdout.flush()
