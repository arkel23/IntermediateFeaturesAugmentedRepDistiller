import time
import wandb
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from distiller.models import model_extractor
from distiller.dataset.loaders import build_dataloaders
from distiller.helper.parser import parse_option_teacher
from distiller.helper.model_tools import save_model
from distiller.helper.optim_tools import return_optimizer_scheduler
from distiller.helper.util import count_params_single, summary_stats
from distiller.helper.loops import train_vanilla as train, validate

def main():
    time_start = time.time()
    best_acc = 0
    max_memory = 0

    opt = parse_option_teacher()

    # dataloader
    train_loader, val_loader, n_cls = build_dataloaders(opt)
    
    # model
    model = model_extractor(opt.model, num_classes=n_cls, layers='last_only')

    # optimizer and scheduler
    optimizer, lr_scheduler = return_optimizer_scheduler(opt, model)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    wandb.init(config=opt)
    wandb.run.name = '{}'.format(opt.model_name)

    # routine
    for epoch in range(1, opt.epochs+1):
        
        lr_scheduler.step(epoch)        
        print("==> Training...Epoch: {} | LR: {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        wandb.log({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss})
        
        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
        wandb.log({'test_acc': test_acc, 'test_loss': test_loss, 'test_acc_top5': test_acc_top5})

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_model(opt, model, epoch, test_acc, mode='best', optimizer=optimizer)
        # regular saving
        if epoch % opt.save_freq == 0:
            save_model(opt, model, epoch, test_acc, mode='epoch', optimizer=optimizer)
        # VRAM memory consumption
        curr_max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
        if curr_max_memory > max_memory:
            max_memory = curr_max_memory
            
    # save last model
    save_model(opt, model, epoch, test_acc, mode='last', optimizer=optimizer)
    
    # summary stats
    time_end = time.time()
    time_total = time_end - time_start
    no_params = count_params_single(model)
    summary_stats(opt.epochs, time_total, best_acc, best_epoch, max_memory, no_params)


if __name__ == '__main__':
    main()
