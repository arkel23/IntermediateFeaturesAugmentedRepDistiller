"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import time

import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_extractor
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.build_dataset import build_dataloader

from helper.util import count_params_module_list, save_model, summary_stats, return_optimizer_scheduler

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss
from ifacrd.criterion_ifacrd import IFACRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--image_size', type=int, default=32, help='image_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--opt', default='sgd', type=str, help='Optimizer (default: "sgd"')
    parser.add_argument('--base_lr', type=float, default=0.2, help='base learning rate to scale based on batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument('--sched', default='warmup_step', type=str, choices=['cosine', 'step', 'warmup_step'],
                        help='LR scheduler (default: "warmup_step"')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--decay_epochs', type=float, default=30, help='epoch interval to decay LR')
    
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['ifacrd', 'kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    
    # IFACRD distillation
    parser.add_argument('--layers', type=str, default='last', choices=['all', 'blocks', 'last'], 
                        help='features from last layers or blocks ends')
    parser.add_argument('--cont_no_l', default=2, type=int, 
                        help='no of layers from teacher to use to build contrastive batch')
    
    parser.add_argument('--rs_no_l', default=1, choices=[1, 2, 3], type=int, 
                        help='no of layers for rescaler mlp')
    parser.add_argument('--rs_hid_dim', default=128, type=int, 
                        help='dimension of rescaler mlp hidden layer space')
    parser.add_argument('--rs_ln', action='store_true', help='Use rescaler mlp with LN instead of BN')
    
    parser.add_argument('--proj_no_l', default=1, choices=[1, 2, 3], type=int, 
                        help='no of layers for projector mlp')
    parser.add_argument('--proj_hid_dim', default=128, type=int, 
                        help='dimension of projector mlp hidden layer space')
    parser.add_argument('--proj_ln', action='store_true', help='Use projector mlp with LN instead of BN')
    
    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set layers argument to blocks when using any method that is not ifacrd
    if opt.distill != 'ifacrd':
        opt.layers = 'default'
        opt.cont_no_l = 0

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.base_lr = opt.base_lr / 5 # base_lr 0.04 and with bs=64 > lr=0.01

    opt.lr = opt.base_lr * (opt.batch_size / 256)
    
    if opt.sched == 'warmup_step' and opt.warmup_epochs == 5:
        opt.warmup_epochs = 150
    
    opt.model_path = './save/student_model'
    
    #iterations = opt.lr_decay_epochs.split(',')
    #opt.lr_decay_epochs = list([])
    #for it in iterations:
    #    opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    if opt.distill == 'ifacrd':
        opt.model_name = 'S{}_T{}_{}_{}_r{}_a{}_b{}_bs{}_blr{}wd{}_temp{}_contl{}{}_rsl{}hd{}ln{}_pjl{}out{}hd{}ln{}_{}'.format(
            opt.model_s, opt.model_t, opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.batch_size, 
            opt.base_lr, opt.weight_decay, opt.nce_t, opt.cont_no_l, opt.layers, opt.rs_no_l, opt.rs_hid_dim, opt.rs_ln, 
            opt.proj_no_l, opt.feat_dim, opt.proj_hid_dim, opt.proj_ln, opt.trial)
    else:
        opt.model_name = 'S{}_T{}_{}_{}_r{}_a{}_b{}_bs{}_blr{}wd{}_temp{}_{}'.format(
            opt.model_s, opt.model_t, opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.batch_size,
            opt.base_lr, opt.weight_decay, opt.nce_t, opt.trial)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    print(opt)
    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls, layers):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_extractor(model_t, num_classes=n_cls, layers=layers)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    time_start = time.time()
    best_acc = 0
    max_memory = 0

    opt = parse_option()
    
    # dataloader
    train_loader, val_loader, n_cls, n_data = build_dataloader(opt, vanilla=False)
    
    # model
    model_t = load_teacher(opt.path_t, n_cls, opt.layers)
    model_s = model_extractor(opt.model_s, num_classes=n_cls, layers=opt.layers)

    data = torch.randn(2, 3, opt.image_size, opt.image_size)
    model_t.eval()
    model_s.eval()
    out_t = model_t(data, classify_only=False)
    out_s = model_s(data, classify_only=False)
    feat_t = out_t[:-1]
    feat_s = out_s[:-1]
    
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'ifacrd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        criterion_kd = IFACRDLoss(opt, model_t)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
        if opt.cont_no_l != 1:
            module_list.append(criterion_kd.rescaler)
            trainable_list.append(criterion_kd.rescaler)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        raise NotImplementedError
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    #optimizer = optim.SGD(trainable_list.parameters(),
    #                      lr=opt.learning_rate,
    #                      momentum=opt.momentum,
    #                      weight_decay=opt.weight_decay)
    optimizer, lr_scheduler = return_optimizer_scheduler(opt, trainable_list)


    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True
        
    wandb.init(config=opt)
    wandb.run.name = '{}'.format(opt.model_name)

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        lr_scheduler.step(epoch)        
        print("==> Training...Epoch: {} | LR: {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        wandb.log({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss})
        
        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
        wandb.log({'test_acc': test_acc, 'test_loss': test_loss, 'test_acc_top5': test_acc_top5})

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_model(opt, model_s, epoch, test_acc, mode='best', vanilla=False)
        # regular saving
        if epoch % opt.save_freq == 0:
            save_model(opt, model_s, epoch, test_acc, mode='epoch', vanilla=False)
        # VRAM memory consumption
        curr_max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
        if curr_max_memory > max_memory:
            max_memory = curr_max_memory
    
    # save last model     
    save_model(opt, model_s, epoch, test_acc, mode='last', vanilla=False)

    # summary stats
    time_end = time.time()
    time_total = time_end - time_start
    
    #if opt.distill == 'abound':
    #    module_list.append(connector)
    no_params_modules = count_params_module_list(module_list)
    no_params_criterion = count_params_module_list(criterion_list)
    no_params = no_params_modules + no_params_criterion
    
    summary_stats(opt.epochs, time_total, best_acc, best_epoch, max_memory, no_params)


if __name__ == '__main__':
    main()
