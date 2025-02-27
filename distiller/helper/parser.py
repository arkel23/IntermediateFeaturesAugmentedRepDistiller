import os
import argparse
import torch

from .model_utils import get_model_name


def parse_common():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('-t', '--trial', type=int,
                        default=0, help='the experiment id')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--print_freq', type=int,
                        default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int,
                        default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size')
    parser.add_argument('--image_size', type=int,
                        default=None, help='image_size')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240,
                        help='number of training epochs')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model on imagenet')

    # optimization
    parser.add_argument('--opt', default='sgd', type=str,
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--base_lr', type=float, default=0.2,
                        help='base learning rate to scale based on batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm (default: None, no clipping)')
    # scheduler
    parser.add_argument('--sched', default='warmup_step', type=str, choices=['cosine', 'step', 'warmup_step'],
                        help='LR scheduler (default: "warmup_step"')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')
    parser.add_argument('--decay_epochs', type=float,
                        default=30, help='epoch interval to decay LR')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'svhn', 'stl10',
                                 'cinic10', 'tinyimagenet', 'imagenet'], help='dataset')
    parser.add_argument('--dataset_path', type=str,
                        default='./data/', help='path to download/read datasets')

    # distributed
    parser.add_argument('--dist_eval', action='store_true',
                        help='validate using dist sampler (else do it on one gpu)')

    return parser


def add_adjust_common_dependent(opt):

    if not opt.image_size:
        if opt.dataset == 'imagenet':
            opt.image_size = 224
        else:
            opt.image_size = 32

    # set different learning rate from these 4 models
    if hasattr(opt, 'model'):
        if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
            opt.base_lr = opt.base_lr / 5  # base_lr 0.04 and with bs=64 > lr=0.01
    elif hasattr(opt, 'model_s'):
        if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
            opt.base_lr = opt.base_lr / 5  # base_lr 0.04 and with bs=64 > lr=0.01

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
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        opt.world_size = torch.distributed.get_world_size()
        opt.rank = torch.distributed.get_rank()

        opt.lr = opt.base_lr * ((opt.world_size * opt.batch_size) / 256)

    return opt


def parse_option_teacher():

    parser = parse_common()
    parser.add_argument('--model', type=str, default='resnet8', choices=[
        'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56',
        'resnet110', 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1',
        'wrn_40_2', 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet18',
        'ResNet34', 'ResNet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2',
        'B_16', 'B_32', 'L_16', 'Bs_16', 'Bss_16'])
    parser = add_vit_args(parser)
    opt = parser.parse_args()

    opt = add_adjust_common_dependent(opt)

    opt.model_name = '{}_{}_is{}_bs{}_blr{}decay{}_tr{}sd{}'.format(
        opt.model, opt.dataset, opt.image_size, opt.batch_size, opt.base_lr,
        opt.weight_decay, opt.trial, opt.seed)

    opt.save_folder = os.path.join('save', 'models', opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    print(opt)
    return opt


def parse_option_linear():

    parser = parse_common()
    parser.add_argument('--sste', action='store_true',
                        help='SS training and evaluation')
    parser.add_argument('--path_model', type=str,
                        default=None, help='model snapshot')
    parser.set_defaults(epochs=100, base_lr=0.4, sched='cosine')
    opt = parser.parse_args()

    if opt.sste:
        with open('sste.txt', 'r') as myfile:
            opt.path_model = myfile.read().splitlines()[-1]
    opt.model = get_model_name(opt.path_model)
    opt = add_adjust_common_dependent(opt)

    opt.model_name = 'linear_{}_{}_is{}_bs{}_blr{}decay{}_tr{}sd{}'.format(
        opt.model, opt.dataset, opt.image_size, opt.batch_size, opt.base_lr,
        opt.weight_decay, opt.trial, opt.seed)

    opt.save_folder = os.path.join('save', 'linear', opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    print(opt)
    return opt


def add_vit_args(parser):
    # default config is based on vit_b16
    parser.add_argument('--classifier', type=str,
                        default=None, choices=('cls', 'pool'))
    # encoder related
    parser.add_argument('--pos_embedding_type', type=str, default='learned',
                        help='positional embedding for encoder, def: learned')
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--intermediate_size', type=int, default=None)
    parser.add_argument('--num_attention_heads', type=int, default=None)
    parser.add_argument('--num_hidden_layers', type=int, default=None)
    parser.add_argument('--encoder_norm', action='store_true',
                        help='norm after encoder')
    # transformers in general
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.0)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--hidden_act', type=str, default=None)
    parser.add_argument('--sd', type=float, default=0.0)
    return parser


def parse_option_student():

    parser = parse_common()
    parser.add_argument('--sste', action='store_true',
                        help='SS training and evaluation')
    # model
    parser.add_argument('--model_s', type=str, default='resnet8', choices=[
        'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56',
        'resnet110', 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1',
        'wrn_40_2', 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet18',
        'ResNet34', 'ResNet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2',
        'B_16', 'B_32', 'L_16', 'Bs_16', 'Bss_16'])
    parser.add_argument('--path_t', type=str, default=None,
                        help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=[
        'ifacrd', 'ifacrdv2', 'kd', 'hint', 'attention', 'similarity', 'correlation',
        'vid', 'crd', 'kdsvd', 'fsp', 'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--init_epochs', type=int, default=30,
                        help='init training for two-stage methods')
    parser.add_argument('-r', '--gamma', type=float,
                        default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float,
                        default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0,
                        help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4,
                        help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128,
                        type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact',
                        type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int,
                        help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float,
                        help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float,
                        help='momentum for non-parametric updates')

    # IFACRD(v2) distillation
    parser.add_argument('--layers', type=str, default='last', choices=['all', 'blocks', 'last'],
                        help='features from last layers or blocks ends')
    parser.add_argument('--cont_no_l', default=1, type=int,
                        help='no of layers from teacher to use to build contrastive batch')

    parser.add_argument('--rs_no_l', default=1, choices=[1, 2, 3], type=int,
                        help='no of layers for rescaler mlp')
    parser.add_argument('--rs_hid_dim', default=128, type=int,
                        help='dimension of rescaler mlp hidden layer space')
    parser.add_argument('--rs_bn', action='store_true',
                        help='Use rescaler mlp with BN')
    parser.add_argument('--rs_ln', action='store_true',
                        help='Use rescaler mlp with LN')
    parser.add_argument('--rs_detach', action='store_false',
                        help='Detach features before passing through rescaler (default: True)')
    parser.add_argument('--rs_no_l_ada', action='store_true',
                        help='Use more layers on Rescaler for shallower layers')
    parser.add_argument('--rs_mixer', action='store_true', help='Use mixer for rescaler')
    parser.add_argument('--rs_transformer', action='store_true', help='Use transformer for rescaler')

    parser.add_argument('--proj_no_l', default=1, choices=[1, 2, 3], type=int,
                        help='no of layers for projector mlp')
    parser.add_argument('--proj_hid_dim', default=128, type=int,
                        help='dimension of projector mlp hidden layer space')
    parser.add_argument('--proj_ind', action='store_true',
                        help='Individual projector for each feature')
    parser.add_argument('--proj_bn', action='store_true',
                        help='Use projector mlp with BN')
    parser.add_argument('--proj_ln', action='store_true',
                        help='Use projector mlp with LN')
    parser.add_argument('--proj_out_norm', action='store_true',
                        help='Use norm at end of projector (simsiam)')

    parser.add_argument('--simclr_aug', action='store_true',
                        help='Use simclr augs')
    parser.add_argument('--sskd', action='store_true',
                        help='KL div on ss module')
    parser.add_argument('--cont_s', type=int, default=0,
                        choices=[0, 1, 2, 3, 4])
    parser.add_argument('--cont_t', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='0 uses no aug, 1 uses aug1, 2 uses aug2, 3 uses aug1 and 2, 4 uses no aug, aug1 and aug2')

    # hint layer
    parser.add_argument('--hint_layer', default=2,
                        type=int, choices=[0, 1, 2, 3, 4])

    parser = add_vit_args(parser)

    opt = parser.parse_args()

    if opt.path_t:
        opt.model_t = get_model_name(opt.path_t)
    else:
        opt.model_t = opt.model_s

    opt = add_adjust_common_dependent(opt)

    opt.rs_no_pool = False
    # set layers argument to default when using any method that is not ifacrd(/v2)
    if opt.distill not in ['ifacrd', 'ifacrdv2']:
        if opt.distill == 'abound':
            opt.layers = 'preact'
        else:
            opt.layers = 'default'
        opt.simclr_aug = False
        opt.sskd = False
        opt.cont_no_l = 0
        opt.cont_s = 0
        opt.cont_t = 0
    else:
        if opt.simclr_aug is False:
            opt.cont_s = 0
            opt.cont_t = 0
        if opt.distill == 'ifacrdv2':
            opt.simclr_aug = True
        else:
            opt.sskd = False
        if opt.rs_mixer or opt.rs_transformer:
            opt.rs_no_pool = True
        opt.distill_ext = f'{opt.distill}_{opt.cont_no_l}_{opt.proj_ind}'

    if opt.sste:
        opt.gamma = 0
        opt.alpha = 0
        opt.sched = 'cosine'

    opt.model_name = 'S{}_T{}_{}_{}_r{}_a{}_b{}_bs{}_blr{}wd{}_temp{}_tr{}sd{}'.format(
            opt.model_s, opt.model_t, opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.batch_size,
            opt.base_lr, opt.weight_decay, opt.nce_t, opt.trial, opt.seed)

    opt.save_folder = os.path.join('save', 'student_model', opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    if opt.sste:
        with open('sste.txt', 'w') as myfile:
            myfile.write(os.path.join(opt.save_folder,
                         '{}_last.pth'.format(opt.model_s)))

    print(opt)
    return opt
