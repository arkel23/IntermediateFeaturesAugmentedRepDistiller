# https://github.com/Spijkervet/SimCLR/blob/master/linear_evaluation.py
# https://github.com/winycg/HSAKD/blob/main/eval_rep.py
from __future__ import print_function

import os
import argparse
import time

import wandb
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_extractor

from dataset.build_dataset import build_dataloader

from helper.util import count_params_single, save_model, summary_stats
from helper.util import return_optimizer_scheduler, get_model_name

from helper.loops import train_vanilla as train, validate, feature_extraction

class LinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(LinearClassifier, self).__init__()
        
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--path_model', type=str, default=None, help='model snapshot')

    # optimization
    parser.add_argument('--opt', default='sgd', type=str, help='Optimizer (default: "sgd"')
    parser.add_argument('--base_lr', type=float, default=0.4, help='base learning rate to scale based on batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument('--sched', default='warmup_step', type=str, choices=['cosine', 'step', 'warmup_step'],
                        help='LR scheduler (default: "warmup_step"')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--decay_epochs', type=float, default=30, help='epoch interval to decay LR')
    
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', 
                        choices=['cifar10', 'cifar100', 'svhn', 'stl10', 'cinic10', 'tinyimagenet', 'imagenet'], help='dataset')
    parser.add_argument('--dataset_path', type=str, default='./data/', help='path to download/read datasets')
    
    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()

    if opt.dataset == 'imagenet':
        opt.image_size = 224
    else:
        opt.image_size = 32
 
    if opt.sched == 'warmup_step' and opt.warmup_epochs == 5:
        opt.warmup_epochs = 150
    
    opt.model = get_model_name(opt.path_model)

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.base_lr = opt.base_lr / 5 # base_lr 0.04 and with bs=64 > lr=0.01
    if opt.model == 'ShuffleV2':
        raise NotImplementedError

    opt.lr = opt.base_lr * (opt.batch_size / 256)

    opt.model_name = '{}_{}_is{}_bs{}_blr{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, 
        opt.image_size, opt.batch_size, opt.base_lr, opt.weight_decay, opt.trial)

    opt.save_folder = os.path.join('save', 'linear', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    print(opt)
    return opt


def load_model(path_model, n_cls, layers):
    print('==> loading model')
    model_name = get_model_name(path_model)
    model = model_extractor(model_name, num_classes=n_cls, layers=layers)
    
    state_dict = torch.load(path_model)['model']
    for key in list(state_dict.keys())[-2:]:
        state_dict.pop(key)
    
    ret = model.load_state_dict(state_dict, strict=False)
    print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys))
    print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys))
    print('==> done')
    return model


def get_features(backbone, train_loader, val_loader, opt):
    train_X, train_y = feature_extraction(train_loader, backbone, opt)
    test_X, test_y = feature_extraction(val_loader, backbone, opt)
    return train_X, train_y, test_X, test_y


def get_features_size(backbone, opt):
    data = torch.randn(2, 3, opt.image_size, opt.image_size)
    with torch.no_grad():
        out = backbone(data, classify_only=False)
    logits_dim = out[-2].shape[-1]
    return logits_dim


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, pin_memory=True)

    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    val_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return train_loader, val_loader


def main():
    time_start = time.time()
    best_acc = 0
    max_memory = 0

    opt = parse_option()
    
    # dataloader
    train_loader, val_loader, n_cls = build_dataloader(opt)
    
    # backbone
    backbone = load_model(opt.path_model, n_cls, 'last_only')
    backbone.eval()
    
    # linear classifier head
    logits_dim = get_features_size(backbone, opt)
    classifier = LinearClassifier(in_features=logits_dim, num_classes=n_cls)
    
    # optimizer and scheduler
    optimizer, lr_scheduler = return_optimizer_scheduler(opt, classifier)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        backbone = backbone.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    wandb.init(config=opt)
    wandb.run.name = '{}'.format(opt.model_name)
    
    (train_X, train_y, test_X, test_y) = get_features(backbone, train_loader, val_loader, opt)

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, opt.batch_size
    )

    for epoch in range(1 , opt.epochs+1):
        
        lr_scheduler.step(epoch)        
        print("==> Training...Epoch: {} | LR: {}".format(epoch, optimizer.param_groups[0]['lr']))
        
        train_acc, train_loss = train(epoch, arr_train_loader, classifier, criterion, optimizer, opt)
        wandb.log({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss})
        
        test_acc, test_acc_top5, test_loss = validate(arr_test_loader, classifier, criterion, opt)
        wandb.log({'test_acc': test_acc, 'test_loss': test_loss, 'test_acc_top5': test_acc_top5})  
    
        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_model(opt, classifier, epoch, test_acc, mode='best', optimizer=optimizer)
        # regular saving
        if epoch % opt.save_freq == 0:
            save_model(opt, classifier, epoch, test_acc, mode='epoch', optimizer=optimizer)
        # VRAM memory consumption
        curr_max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
        if curr_max_memory > max_memory:
            max_memory = curr_max_memory

    # save last model
    save_model(opt, classifier, epoch, test_acc, mode='last', optimizer=optimizer)
    
    # summary stats
    time_end = time.time()
    time_total = time_end - time_start
    no_params = count_params_single(classifier)
    summary_stats(opt.epochs, time_total, best_acc, best_epoch, max_memory, no_params)
    

if __name__ == '__main__':
    main()