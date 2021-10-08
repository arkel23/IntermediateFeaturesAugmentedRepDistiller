import os
import argparse
import torch

from .model_utils import get_model_name

def parse_common():
    
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--opt', default='sgd', type=str, help='Optimizer (default: "sgd"')
    parser.add_argument('--base_lr', type=float, default=0.2, help='base learning rate to scale based on batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, help='Clip gradient norm (default: None, no clipping)')
    # scheduler
    parser.add_argument('--sched', default='warmup_step', type=str, choices=['cosine', 'step', 'warmup_step'],
                        help='LR scheduler (default: "warmup_step"')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--decay_epochs', type=float, default=30, help='epoch interval to decay LR')
    
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', 
                        choices=['cifar10', 'cifar100', 'svhn', 'stl10', 'cinic10', 'tinyimagenet', 'imagenet'], help='dataset')
    parser.add_argument('--dataset_path', type=str, default='./data/', help='path to download/read datasets')
    
    return parser


def add_adjust_common_dependent(opt):
    
    if opt.dataset == 'imagenet':
        opt.image_size = 224
    else:
        opt.image_size = 32
    
    # set different learning rate from these 4 models
    if hasattr(opt, 'model'):
        if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
            opt.base_lr = opt.base_lr / 5 # base_lr 0.04 and with bs=64 > lr=0.01
    elif hasattr(opt, 'model_s'):
        if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
            opt.base_lr = opt.base_lr / 5 # base_lr 0.04 and with bs=64 > lr=0.01
        
    opt.lr = opt.base_lr * (opt.batch_size / 256)
    
    if opt.sched == 'warmup_step' and opt.warmup_epochs == 5:
        opt.warmup_epochs = 150
        
    # distributed
    opt.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    opt.world_size = 1
    opt.rank = 0  # global rank
    opt.local_rank = 0
    
    opt.distributed = False
    if 'WORLD_SIZE' in os.environ:
        opt.local_rank = int(os.environ['LOCAL_RANK'])
        opt.distributed = int(os.environ['WORLD_SIZE']) > 1
    
    if opt.distributed:
        opt.device = 'cuda:%d' % opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        opt.world_size = torch.distributed.get_world_size()
        opt.rank = torch.distributed.get_rank()
        
        opt.lr = opt.base_lr * ((opt.world_size * opt.batch_size) / 256)
    
    return opt
 
 
def parse_option_teacher():
    
    parser = parse_common()
    parser.add_argument('--model', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet18', 'ResNet34', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    opt = parser.parse_args()
    
    opt = add_adjust_common_dependent(opt)
    
    opt.model_name = '{}_{}_is{}_bs{}_blr{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, 
        opt.image_size, opt.batch_size, opt.base_lr, opt.weight_decay, opt.trial)
    
    opt.save_folder = os.path.join('save', 'models', opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    print(opt)
    return opt


def parse_option_linear():
    
    parser = parse_common()
    parser.add_argument('--path_model', type=str, default=None, help='model snapshot')
    parser.set_defaults(epochs=100, base_lr=0.4, sched='cosine')
    opt = parser.parse_args()

    opt.model = get_model_name(opt.path_model)
    opt = add_adjust_common_dependent(opt)

    opt.model_name = 'linear_{}_{}_is{}_bs{}_blr{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, 
        opt.image_size, opt.batch_size, opt.base_lr, opt.weight_decay, opt.trial)

    opt.save_folder = os.path.join('save', 'linear', opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    print(opt)
    return opt


def parse_option_student():
    
    parser = parse_common()
    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet18', 'ResNet34', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['ifacrd', 'kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])    
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')
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

    opt.model_t = get_model_name(opt.path_t)
    opt = add_adjust_common_dependent(opt)
    # set layers argument to blocks when using any method that is not ifacrd
    if opt.distill != 'ifacrd':
        if opt.distill == 'abound':
            opt.layers = 'preact'
        else:
            opt.layers = 'default'
        opt.cont_no_l = 0

    if opt.distill == 'ifacrd':
        opt.model_name = 'S{}_T{}_{}_{}_r{}_a{}_b{}_bs{}_blr{}wd{}_temp{}_contl{}{}_rsl{}hd{}ln{}_pjl{}out{}hd{}ln{}_{}'.format(
            opt.model_s, opt.model_t, opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.batch_size, 
            opt.base_lr, opt.weight_decay, opt.nce_t, opt.cont_no_l, opt.layers, opt.rs_no_l, opt.rs_hid_dim, opt.rs_ln, 
            opt.proj_no_l, opt.feat_dim, opt.proj_hid_dim, opt.proj_ln, opt.trial)
    else:
        opt.model_name = 'S{}_T{}_{}_{}_r{}_a{}_b{}_bs{}_blr{}wd{}_temp{}_{}'.format(
            opt.model_s, opt.model_t, opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.batch_size,
            opt.base_lr, opt.weight_decay, opt.nce_t, opt.trial)

    opt.save_folder = os.path.join('save', 'student_model', opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    print(opt)
    return opt



