from __future__ import print_function
from builtins import input
import argparse
import importlib
import os
import sys
import time
import shutil
import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from segmentation.loss.loss_function import Dice_Loss
from segmentation.utils.train_tools import *
from segmentation.utils.vseg_dataset  import SegmentationDataset

def worker_init(worker_idx):
    """
    The worker initialization function takes the worker id (an int in "[0,
    num_workers - 1]") as input and does random seed initialization for each
    worker.
    :param worker_idx: The worker index.
    :return: None.
    """
    MAX_INT = sys.maxsize
    worker_seed = np.random.randint(int(np.sqrt(MAX_INT))) + worker_idx
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)


def save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, num_modality):
    """ save model and parameters into a checkpoint file (.pth)

    :param net: the network object
    :param opt: the optimizer object
    :param epoch_idx: the epoch index
    :param batch_idx: the batch index
    :param cfg: the configuration object
    :param config_file: the configuration file path
    :param max_stride: the maximum stride of network
    :param num_modality: the number of image modalities
    :return: None
    """
    chk_folder = os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx))
    if not os.path.isdir(chk_folder):
        os.makedirs(chk_folder)

    filename = os.path.join(chk_folder, 'params.pth')
    opt_filename = os.path.join(chk_folder, 'optimizer.pth')

    state = {'epoch':             epoch_idx,
             'batch':             batch_idx,
             'net':               cfg.net.name,
             'max_stride':        max_stride,
             'state_dict':        net.state_dict(),
             'spacing':           cfg.dataset.spacing,
             'interpolation':     cfg.dataset.interpolation,
             'pad_t':             cfg.dataset.pad_t,
             'default_values':    cfg.dataset.default_values,
             'in_channels':       num_modality,
             'out_channels':      cfg.dataset.num_classes,
             'crop_normalizers':  [normalizer.to_dict() for normalizer in cfg.dataset.crop_normalizers]}

    # save python check point
    torch.save(state, filename)

    # save python optimizer state
    torch.save(opt.state_dict(), opt_filename)

    # save template parameter ini file
    ini_file = os.path.join(os.path.dirname(__file__), 'config', 'params.ini')
    shutil.copy(ini_file, os.path.join(cfg.general.save_dir, 'params.ini'))

    # copy config file
    shutil.copy(config_file, os.path.join(chk_folder, 'config.py'))


def load_checkpoint(epoch_idx, net, opt, save_dir):
    """ load network parameters from directory

    :param epoch_idx: the epoch idx of model to load
    :param net: the network object
    :param opt: the optimizer object
    :param save_dir: the save directory
    :return: loaded epoch index, loaded batch index
    """
    # load network parameters
    chk_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'params.pth')
    assert os.path.isfile(chk_file), 'checkpoint file not found: {}'.format(chk_file)

    state = torch.load(chk_file)
    net.load_state_dict(state['state_dict'])

    # load optimizer state
    opt_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'optimizer.pth')
    assert os.path.isfile(opt_file), 'optimizer file not found: {}'.format(chk_file)

    opt_state = torch.load(opt_file)
    
    loading = True
    if loading:
        print("LLLLLLLLLLLOADING OPTIMIZER  !!!")
        opt.load_state_dict(opt_state)
    else:
        print("NOOOOOOOOOO LOADING OPTIMIZER  !!!")

    return state['epoch'], state['batch']

def train(config_file, msg_queue=None):
    """ volumetric segmentation training engine

    :param config_file: the input configuration file
    :param msg_queue: message queue to export runtime message to integrated system
    :return: None
    """
    assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_module_from_disk(config_file)
    cfg = cfg.cfg

    # convert to absolute path since cfg uses relative path
    root_dir = os.path.dirname(config_file)
    cfg.general.save_dir = os.path.join(root_dir, cfg.general.save_dir)

    # control randomness during training
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed(cfg.general.seed)

    # clean the existing folder if not continue training
    if cfg.general.resume_epoch < 0 and os.path.isdir(cfg.general.save_dir):
        sys.stdout.write("Found non-empty save dir.\n"
                         "Type 'yes' to delete, 'no' to continue: ")
        choice = input().lower()
        if choice == 'yes':
            shutil.rmtree(cfg.general.save_dir)
        elif choice == 'no':
            pass
        else:
            raise ValueError("Please type either 'yes' or 'no'!")

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'vseg')

    # enable CUDNN
    cudnn.benchmark = True

    # dataset
    dataset = SegmentationDataset(
                csv_path=cfg.general.csv_path,
                crop_size=cfg.dataset.crop_size,      
                crop_spacing=cfg.dataset.spacing, 
                window_center=cfg.dataset.window_center, 
                window_width=cfg.dataset.window_width,
    )

    sampler = EpochConcateSampler(dataset, cfg.train.epochs)

    data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batchsize,
                             num_workers=cfg.train.num_threads, pin_memory=True, worker_init_fn=worker_init)

    # define network
    gpu_ids = list(range(cfg.general.num_gpus))

    net_module = importlib.import_module('segmentation.network.' + cfg.net.name)

    net = net_module.SegmentationNet(cfg.dataset.num_modality, cfg.dataset.num_classes)
    max_stride = net.max_stride()
    net_module.vnet_kaiming_init(net)
    net = nn.parallel.DataParallel(net, device_ids=gpu_ids)
    net = net.cuda()

    assert np.all(np.array(cfg.dataset.crop_size) % np.array(max_stride) == 0), 'crop size not divisible by max stride'

    # define loss function
    loss_func = Dice_Loss(cfg.train.weight_target, cfg.train.weight_background)

    # training optimizer
    opt = getattr(torch.optim, cfg.train.optimizer.name)(
        [{'params': net.parameters(), 'initial_lr': cfg.train.lr}],
        lr=cfg.train.lr, **cfg.train.optimizer.params
    )

    # load checkpoint if resume epoch > 0
    if cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_start = load_checkpoint(cfg.general.resume_epoch, net, opt, cfg.general.save_dir)
    else:
        last_save_epoch, batch_start = 0, 0

    scheduler = getattr(torch.optim.lr_scheduler, cfg.train.lr_scheduler.name+"LR")(
        optimizer=opt, **cfg.train.lr_scheduler.params)

    batch_idx = batch_start
    if cfg.general.clear_start_epoch:
        batch_idx = 0
    data_iter = iter(data_loader)

    # loop over batches
    for i in range(len(data_loader)):

        begin_t = time.time()

        crops, masks, weight = next(data_iter)

        crops, masks, weight = crops.cuda(), masks.cuda(), weight.cuda()

        # clear previous gradients
        opt.zero_grad()

        # network forward
        outputs = net(crops)

        # the epoch idx of model
        epoch_idx = batch_idx * cfg.train.batchsize // len(dataset)
        train_loss, _  = loss_func(outputs, masks, weight)

        # backward propagation
        train_loss.backward()

        # update weights
        opt.step()

        if epoch_idx != scheduler.last_epoch:
            scheduler.step(epoch=epoch_idx)
        
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / cfg.train.batchsize

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol, time: {:.6f}'
        msg = msg.format(epoch_idx, batch_idx, train_loss.item(), sample_duration, scheduler.get_last_lr()[0])
        logger.info(msg)
        if msg_queue is not None:
            msg_queue.put(msg)
        
        if (batch_idx + 1) % cfg.train.plot_snapshot == 0:
            train_loss_plot_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
            plot_loss(log_file, train_loss_plot_file, name='train_loss',
                      display='Training Loss ({})'.format(cfg.loss.name))

        if epoch_idx != 0 and (epoch_idx % cfg.train.save_epochs == 0):

            if last_save_epoch != epoch_idx:
                save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, dataset.num_modality())
                last_save_epoch = epoch_idx


def main():

    long_description = "Segmentation Train Engine"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', nargs='?',
                        default="/mnt/maui/CTA_AorticDissection/project/home/litong/reconstruction/paper_code/occ_diffusion_segmentation/segmentation/config/config.py",
                        help='volumetric segmentation train config file')
    args = parser.parse_args()

    train(args.input)


if __name__ == '__main__':
    main()
