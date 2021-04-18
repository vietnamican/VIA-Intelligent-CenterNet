import os
import os.path as osp
import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import load_data, load_trainer, load_model 

pl.seed_everything(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument("--train-image-dir", type=str,
                        help='Path to training image folder', default='via-trafficsign/images/train')
    parser.add_argument("--train-label-dir", type=str,
                        help='Path to training image label folder', default='via-trafficsign/labels/train')
    parser.add_argument("--val-image-dir", type=str,
                        help='Path to validation image folder', default='via-trafficsign/images/val')
    parser.add_argument("--val-label-dir", type=str,
                        help='Path to validation image label folder', default='via-trafficsign/labels/val')
    parser.add_argument(
        "--device", type=str, help="Choose what device to train, one of the: ['cpu', 'gpu', 'tpu'], tpu is unavailable now", default='cpu')
    parser.add_argument(
        '--logdir', type=str, help='Choose logdir for tensorboard logger', default='traffic_logs/training')
    parser.add_argument('--epochs', type=int,
                        help='max epoch for training', default=90)
    parser.add_argument('--batch-size', type=int,
                        help='batch size for training, recommended greater than or equal to 16', default=2)
    parser.add_argument('--num-workers', type=int,
                        help='number of workers for loading data', default=4)
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint to load from', default=None)
    parser.add_argument(
        '--backbone', type=str, help="Choose backbone for centernet, one of the ['mobilenet', 'vgg']", default='vgg')

    args = parser.parse_args()

    _, trainloader, _, valloader = load_data(args)
    model = load_model(args)
    trainer = load_trainer(args)
    trainer.fit(model, trainloader, valloader)
