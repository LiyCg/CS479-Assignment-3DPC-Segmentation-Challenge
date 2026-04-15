'''
    this code is implemented based on PointGroup train.py
'''

import torch, time, sys, os, random
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# scripts from PointGroup
from util.config import cfg
from util.log import logger
import util.utils as utils

# ours
import train_dataset
from loss import compute_loss

def init():
    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)

# def train_epoch(train_loader, model, model_fn, optimizer, epoch):
def train_epoch(train_loader, model, optimizer, epoch, device, use_cuda):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    
    model.train()
    start_epoch = time.time()
    end = time.time()
    
    # adjust learning rate (once per epoch)
    utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)
    
    # dictsave epoch-average losses
    epoch_loss = {k: 0.0 for k in list(cfg.loss.keys()) + ['total']}
    
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        # torch.cuda.empty_cache()

        # parse batch data 
        features = batch['features'].to(device).float()                # [B, 9, N_points]
        instance_labels = batch['instance_labels'].to(device).long()   # [B, N_points]
        sem_labels = batch['sem_labels'].to(device).long()             # [B, N_points]

        # forward
        out = model(features)                                          # [B, out_channel, N_points] 
        
        # convert to semantic labels  # ! [CHANGE HERE] modify 'pred' after implementing model 
        gt = {'instance_labels': instance_labels, 'sem_labels': sem_labels}
        pred = {'sem_labels': out}  
        
        # compute loss 
        losses = compute_loss(cfg.loss, pred, gt)
        loss = losses['total']

        # accumulate epoch loss
        for k, v in losses.items():
            epoch_loss[k] += v.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        
        running_loss = epoch_loss['total'] / (i + 1)

        sys.stdout.write(
            "epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f}) "
            "data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {}\n".format(
                epoch,
                cfg.epochs,
                i + 1,
                len(train_loader),
                loss.item(),
                running_loss,
                data_time.val,
                data_time.avg,
                iter_time.val,
                iter_time.avg,
                remain_time
            )
        )
        if i == len(train_loader) - 1:
            print()

    # average losses over epoch
    for k in epoch_loss:
        epoch_loss[k] /= len(train_loader)

    logger.info(
        "epoch: {}/{}, train loss: {:.4f}, time: {:.2f}s".format(
            epoch, cfg.epochs, epoch_loss['total'], time.time() - start_epoch
        )
    )

    if epoch % cfg.save_per == 0 or (epoch == cfg.epochs):
        utils.checkpoint_save(
            model,
            cfg.exp_path,
            cfg.exp_name,
            epoch,
            cfg.save_per,
            use_cuda
        )

    # tensorboard logging
    for k, v in epoch_loss.items():
        writer.add_scalar(k + '_train', v, epoch)


def eval_epoch(val_loader, model, epoch, device):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    model.eval()
    start_epoch = time.time()

    epoch_loss = {k: 0.0 for k in list(cfg.loss.keys()) + ['total']}

    with torch.no_grad():
        
        for i, batch in enumerate(val_loader):

            # parse batch data 
            features = batch['features'].to(device).float()                 # [B, 9, N_points]
            instance_labels = batch['instance_labels'].to(device).long()    # [B, N_points]
            sem_labels = batch['sem_labels'].to(device).long()              # [B, N_points]

            # forward
            out = model(features)                                          # [B, out_channel, N_points] 
            
            # convert to semantic labels  # ! [CHANGE HERE] modify 'pred' after implementing model 
            gt = {'instance_labels': instance_labels, 'sem_labels': sem_labels}
            pred = {'sem_labels': out}  
            
            # compute loss 
            losses = compute_loss(cfg.loss, pred, gt)
            loss = losses['total']
                        
            # accumulate epoch loss
            for k, v in losses.items():
                epoch_loss[k] += v.item()

            running_loss = epoch_loss['total'] / (i + 1)

            sys.stdout.write(
                "\riter: {}/{} loss: {:.4f}({:.4f})".format(
                    i + 1,
                    len(val_loader),
                    loss.item(),
                    running_loss
                )
            )
            if i == len(val_loader) - 1:
                print()

    # average losses over epoch
    for k in epoch_loss:
        epoch_loss[k] /= len(val_loader)

    logger.info(
        "epoch: {}/{}, val loss: {:.4f}, time: {:.2f}s".format(
            epoch, cfg.epochs, epoch_loss['total'], time.time() - start_epoch
        )
    )

    for k, v in epoch_loss.items():
        writer.add_scalar(k + '_eval', v, epoch)

if __name__ == '__main__':

    # initialize 
    init()
    if cfg.save_per < 1 or cfg.val_per < 1:
        raise ValueError("save_per and val_per must be >= 1")

    # get name of current model and data 
    exp_path = cfg.exp_path.split('/')
    model_name = exp_path[-2]
    data_name = exp_path[-3]

    # model
    logger.info('=> creating model ...')
    
    # ! [CHANGE HERE] Load our model 
    if model_name == 'pointgroup':
        from model import DummyModel as Network 
    else:
        raise ValueError("Error: no model - {}".format(model_name))

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = Network(cfg).to(device)

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    # optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        raise ValueError("Unsupported optimizer: {}".format(cfg.optim))
    
    # dataset
    data_dir = os.path.join(cfg.data_root, cfg.dataset)     
    train_data = train_dataset.NubzukiTrainDataset(data_dir=data_dir, num_points=cfg.max_npoint, split='train')
    val_data = train_dataset.NubzukiTrainDataset(data_dir=data_dir, num_points=cfg.max_npoint, split='val')
    
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.train_workers)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.train_workers)

    # resume from the latest epoch, or specify the epoch to restore
    start_epoch = utils.checkpoint_restore(model, cfg.exp_path, cfg.exp_name, use_cuda)

    # train and validation
    for epoch in tqdm(range(start_epoch, cfg.epochs + 1)):
        train_epoch(train_loader, model, optimizer, epoch, device, use_cuda)

        if (epoch % cfg.val_per == 0) or (epoch == cfg.epochs):
            eval_epoch(val_loader, model, epoch, device)