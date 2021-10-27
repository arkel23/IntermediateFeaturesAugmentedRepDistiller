# https://github.com/rwightman/pytorch-image-models/blob/b544ad4d3fcd02057ab9f43b118290f2a089566f/timm/utils/distributed.py#L11
# https://github.com/rwightman/pytorch-image-models/blob/master/train.py
from __future__ import print_function, division

import sys
import time
import torch
import numpy as np

from .misc_utils import AverageMeter, accuracy
from .dist_utils import reduce_tensor, distribute_bn


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            if opt.distributed:
                reduced_loss = reduce_tensor(loss.data, opt.world_size)
                acc1 = reduce_tensor(acc1, opt.world_size)
                acc5 = reduce_tensor(acc5, opt.world_size)
            else:
                reduced_loss = loss.data
                
            losses.update(reduced_loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            
            if opt.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()
    
    if opt.local_rank == 0:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
        
    if opt.distributed:
        distribute_bn(model, opt.world_size, True)        

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()
    elif opt.distill == 'ifacrdv2':
        if opt.sskd:
            module_list[1].eval()
            module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_div = AverageMeter()
    losses_kd = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        elif opt.distill == 'ifacrdv2' or (
            opt.simclr_aug and opt.distill == 'ifacrd'
        ):
            (input, input_aug1, input_aug2), target, index = data
            bs = input.size(0)
            if opt.cont_s in [3, 4] or opt.distill == 'ifacrdv2':
                input_s = torch.cat([input, input_aug1, input_aug2], dim=0)
            elif opt.cont_s == 1:
                input_s = torch.cat([input, input_aug1], dim=0)
            elif opt.cont_s == 2:
                input_s = torch.cat([input, input_aug2], dim=0)
            else:
                input_s = input
                
            if opt.cont_t in [3, 4] or (opt.distill == 'ifacrdv2' and opt.sskd):
                input_t = torch.cat([input, input_aug1, input_aug2], dim=0)
            elif opt.cont_t == 1:
                input_t = torch.cat([input, input_aug1], dim=0)
            elif opt.cont_t == 2:
                input_t = torch.cat([input, input_aug2], dim=0)
            else:
                input_t = input
            
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
            elif opt.distill == 'ifacrdv2' or (
                opt.simclr_aug and opt.distill == 'ifacrd' and (
                    opt.cont_s != 0 or opt.cont_t != 0
                )    
            ):
                #input_aug1 = input_aug1.cuda()
                #input_aug2 = input_aug2.cuda()
                input_s = input_s.cuda()
                input_t = input_t.cuda()
        
        # ===================forward=====================
        if opt.distill == 'ifacrdv2' or (
            opt.simclr_aug and opt.distill == 'ifacrd' and opt.cont_s != 0    
        ):
            out_s = model_s(input_s, classify_only=False)
            feat_s = out_s[:-1]
            logit_s = out_s[-1]
            if opt.cont_s in [3, 4] or opt.distill == 'ifacrdv2':
                feat_s = [torch.split(t, [bs, bs, bs], dim=0) for t in feat_s]
                feat_s, feat_s_aug1, feat_s_aug2 = zip(*feat_s)
                logit_s, _, _ = torch.split(logit_s, [bs, bs, bs], dim=0)
            else:
                feat_s = [torch.split(t, [bs, bs], dim=0) for t in feat_s]
                feat_s, feat_s_aug = zip(*feat_s)
                logit_s, _ = torch.split(logit_s, [bs, bs], dim=0)
        else:
            out_s = model_s(input, classify_only=False)
            feat_s = out_s[:-1]
            logit_s = out_s[-1]
        '''
        if opt.distill in ['ifacrd', 'ifacrdv2'] and opt.simclr_aug:
            if opt.distill == 'ifacrdv2':
                out_s_aug1 = model_s(input_aug1, classify_only=False)
                feat_s_aug1 = out_s_aug1[:-1]
                out_s_aug2 = model_s(input_aug2, classify_only=False)
                feat_s_aug2 = out_s_aug2[:-1]
            elif opt.cont_s == 1:
                out_s_aug1 = model_s(input_aug1, classify_only=False)
                feat_s_aug1 = out_s_aug1[:-1]
            elif opt.cont_s == 2:
                out_s_aug2 = model_s(input_aug2, classify_only=False)
                feat_s_aug2 = out_s_aug2[:-1]
        '''
        with torch.no_grad():
            if (opt.distill == 'ifacrdv2' and opt.sskd) or (
                opt.simclr_aug and opt.distill == 'ifacrd' and opt.cont_t != 0    
            ):
                out_t = model_t(input_t, classify_only=False)
                feat_t = out_t[:-1]
                logit_t = out_t[-1]
                if opt.distill == 'ifacrdv2' or opt.cont_t in [3, 4]:
                    feat_t = [torch.split(t, [bs, bs, bs], dim=0) for t in feat_t]
                    feat_t, feat_t_aug1, feat_t_aug2 = zip(*feat_t)
                    logit_t, _, _ = torch.split(logit_t, [bs, bs, bs], dim=0)
                else:
                    feat_t = [torch.split(t, [bs, bs], dim=0) for t in feat_t]
                    feat_t, feat_t_aug = zip(*feat_t)
                    logit_t, _ = torch.split(logit_t, [bs, bs], dim=0)
            else:
                out_t = model_t(input, classify_only=False)
                feat_t = out_t[:-1]
                logit_t = out_t[-1]
                if opt.distill not in ['ifacrd', 'ifacrdv2']:
                    feat_t = [f.detach() for f in feat_t]
            '''            
            if opt.distill in ['ifacrd', 'ifacrdv2'] and opt.simclr_aug:
                if opt.cont_t in [3, 4] or \
                    (opt.distill == 'ifacrdv2' and opt.sskd):
                    out_t_aug1 = model_t(input_aug1, classify_only=False)
                    feat_t_aug1 = out_t_aug1[:-1]
                    out_t_aug2 = model_t(input_aug2, classify_only=False)
                    feat_t_aug2 = out_t_aug2[:-1]
                elif opt.cont_t == 1:
                    out_t_aug1 = model_t(input_aug1, classify_only=False)
                    feat_t_aug1 = out_t_aug1[:-1]                         
                elif opt.cont_t == 2:
                    out_t_aug2 = model_t(input_aug2, classify_only=False)
                    feat_t_aug2 = out_t_aug2[:-1]
            '''    
        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = torch.tensor(0)
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'ifacrd':
            if opt.simclr_aug:
                if opt.cont_s == 4:
                    f_s = [feat_s[-1], feat_s_aug1[-1], feat_s_aug2[-1]]
                elif opt.cont_s == 3:
                    f_s = [feat_s_aug1[-1], feat_s_aug2[-1]]
                elif opt.cont_s == 0:
                    f_s = feat_s[-1]
                else:
                    f_s = feat_s_aug[-1]
                
                if opt.cont_t == 4:
                    f_t = [feat_t, feat_t_aug1, feat_t_aug2]
                elif opt.cont_t == 3:
                    f_t = [feat_t_aug1, feat_t_aug2]
                elif opt.cont_t == 0:
                    f_t = feat_t
                else:
                    f_t = feat_t_aug
            else:
                f_s = feat_s[-1]
                f_t = feat_t            
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'ifacrdv2':
            if opt.sskd:
                logits_ss_t = criterion_kd(
                    feat_t_aug1, feat_t_aug2, module_list[1], module_list[2])
                logits_ss_s = criterion_kd(
                    feat_s_aug1, feat_s_aug2, module_list[3], module_list[4])
                loss_kd = criterion_div(logits_ss_t, logits_ss_s)
            else:
                loss_kd = criterion_kd(
                    feat_s_aug1, feat_s_aug2, module_list[1], module_list[2])
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = torch.tensor(0)
            #g_s = module_list[1](feat_s[1:-1])
            #g_t = feat_t[1:-1]
            #loss_group = criterion_kd(g_s, g_t)
            #loss_kd = sum(loss_group)
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
            #loss_group = criterion_kd(feat_s[:-1], feat_t[:-1])
            #loss_kd = sum(loss_group)
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        torch.cuda.synchronize()        
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            if opt.distributed:
                reduced_loss = reduce_tensor(loss.data, opt.world_size)
                loss_cls = reduce_tensor(loss_cls.data, opt.world_size)
                loss_div = reduce_tensor(loss_div.data, opt.world_size)
                loss_kd = reduce_tensor(loss_kd.data, opt.world_size)
                acc1 = reduce_tensor(acc1, opt.world_size)
                acc5 = reduce_tensor(acc5, opt.world_size)
            else:
                reduced_loss = loss.data
                loss_cls = loss_cls.data
                loss_div = loss_div.data
                loss_kd = loss_kd.data
            
            losses.update(reduced_loss.item(), input.size(0))
            losses_cls.update(loss_cls.item(), input.size(0))
            losses_div.update(loss_div.item(), input.size(0))
            losses_kd.update(loss_kd.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            
            if opt.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Loss cls {losses_cls.val:.4f} ({losses_cls.avg:.4f})\t'
                    'Loss div {losses_div.val:.4f} ({losses_div.avg:.4f})\t'
                    'Loss kd {losses_kd.val:.4f} ({losses_kd.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, losses_cls=losses_cls,
                    losses_div=losses_div, losses_kd=losses_kd,
                    top1=top1, top5=top5))
                sys.stdout.flush()
    if opt.local_rank == 0:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    if opt.distributed:
        distribute_bn(module_list, opt.world_size, True)        

    return top1.avg, losses.avg, losses_cls.avg, losses_div.avg, losses_kd.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            torch.cuda.synchronize()
            
            if opt.distributed:
                reduced_loss = reduce_tensor(loss.data, opt.world_size)
                acc1 = reduce_tensor(acc1, opt.world_size)
                acc5 = reduce_tensor(acc5, opt.world_size)
            else:
                reduced_loss = loss.data
            
            losses.update(reduced_loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0 and opt.local_rank == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
        
        if opt.local_rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def feature_extraction(loader, backbone, opt):
    feature_vector = []
    labels_vector = []
    for idx, (x, y) in enumerate(loader):
        if torch.cuda.is_available():
            x = x.cuda()

        # get encoding
        with torch.no_grad():
            output = backbone(x, classify_only=False)
        features = output[-2].detach()

        feature_vector.extend(features.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if idx % opt.print_freq == 0 and opt.local_rank == 0:
            print(f"Step [{idx}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    if opt.local_rank == 0:
        print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector
