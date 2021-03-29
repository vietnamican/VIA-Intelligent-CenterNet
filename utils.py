from models import backbone
import os
import os.path as osp
import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import Config as cfg
from models.model import Model
from models.backbone.vgg import VGG
from models.backbone.mobilenetv2 import MobileNetV2, load_mobile_net
from datasets import TrafficDataset


def load_data(args, val_only=False):
    if not val_only:
        train_image_dir = args.train_image_dir
        train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    # Data Setup
    if not val_only:
        traindataset = TrafficDataset(
            train_image_dir, train_label_dir, 'train')
        trainloader = DataLoader(traindataset, batch_size=batch_size,
                                 pin_memory=True, num_workers=num_workers)

    valdataset = TrafficDataset(val_image_dir, val_label_dir, 'val')
    valloader = DataLoader(valdataset, batch_size=batch_size,
                           pin_memory=True, num_workers=num_workers)
    if not val_only:
        return traindataset, trainloader, valdataset, valloader
    return valdataset, valloader


def load_model(args):
    backbone = args.backbone
    if backbone == 'vgg':
        base = VGG()
    else:
        base = load_mobile_net(pretrained=True)
    net = Model(base)
    return net


def load_trainer(args):
    logdir = args.logdir
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name=logdir,
    )

    loss_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='',
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        mode='min',
    )
    callbacks = [loss_callback]
    device = args.device
    max_epochs = args.epochs
    resume_from_checkpoint = args.checkpoint
    if device == 'tpu':
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            tpu_cores=8,
            resume_from_checkpoint=resume_from_checkpoint
        )
    elif device == 'gpu':
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            gpus=1,
            resume_from_checkpoint=resume_from_checkpoint
        )
    else:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            resume_from_checkpoint=resume_from_checkpoint
        )

    return trainer
