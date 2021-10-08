import time
import wandb
import torch

from distiller.models import model_extractor
from distiller.dataset.loaders import build_dataloaders
from distiller.helper.parser import parse_option_teacher
from distiller.helper.misc_utils import count_params_single, random_seed, summary_stats
from distiller.helper.model_utils import save_model
from distiller.helper.optim_utils import return_optimizer_scheduler
from distiller.helper.loops import train_vanilla as train, validate

def main():
    time_start = time.time()
    best_acc = 0
    max_memory = 0

    opt = parse_option_teacher()
    random_seed(opt.seed, opt.rank)

    # dataloader
    train_loader, val_loader, n_cls = build_dataloaders(opt)

    # model and criterion
    model = model_extractor(opt.model, num_classes=n_cls, layers='last_only')
    model.to(opt.device)
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)

    # optimizer and scheduler
    optimizer, lr_scheduler = return_optimizer_scheduler(opt, model)

    if opt.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if opt.local_rank == 0:
        wandb.init(config=opt)
        wandb.run.name = '{}'.format(opt.model_name)

    # routine
    for epoch in range(1, opt.epochs+1):
        if opt.distributed:
            train_loader.sampler.set_epoch(epoch)
        lr_scheduler.step(epoch)
        
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        test_acc, test_loss = validate(val_loader, model, criterion, opt)

        if opt.local_rank == 0:
            print("==> Training...Epoch: {} | LR: {}".format(epoch, optimizer.param_groups[0]['lr']))
            wandb.log({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss, 
                       'test_acc': test_acc, 'test_loss': test_loss})
            
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

    if opt.local_rank == 0:
        # save last model
        save_model(opt, model, epoch, test_acc, mode='last', optimizer=optimizer)

        # summary stats
        time_end = time.time()
        time_total = time_end - time_start
        no_params = count_params_single(model)
        summary_stats(opt.epochs, time_total, best_acc, best_epoch, max_memory, no_params)


if __name__ == '__main__':
    main()
