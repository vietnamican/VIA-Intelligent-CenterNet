import os
from tqdm import tqdm
import argparse

import cv2
import torch
import numpy as np
import pytorch_lightning as pl

# local imports
from config import Config as cfg
from models.backbone.vgg import VGG
from models.model import Model
from utils import load_data, load_model as _load_model, detect, decode, visualize


def load_model(args):
    net = _load_model(args)
    checkpoint_path = args.checkpoint
    device = args.device
    if device == 'cpu':
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    net.eval()
    net.load_state_dict(state_dict)
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument("--val-image-dir", type=str,
                        help='Path to validation image folder', default='via-trafficsign/images/val')
    parser.add_argument("--val-label-dir", type=str,
                        help='Path to validation image label folder', default='via-trafficsign/labels/val')
    parser.add_argument(
        "--device", type=str, help="Choose what device to train, one of the: ['cpu', 'gpu', 'tpu'], tpu is unavailable now", default='cpu')
    parser.add_argument(
        '--logdir', type=str, help='Choose logdir for tensorboard logger', default='traffic_logs/training')
    parser.add_argument('--batch-size', type=int,
                        help='batch size for training, recommended greater than or equal to 16', default=1)
    parser.add_argument('--num-workers', type=int,
                        help='number of workers for loading data', default=4)
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint to load from', required=True)

    parser.add_argument('--outdir', type=str,
                        help='Directory to save output result', required=True)
    parser.add_argument(
        '--backbone', type=str, help="Choose backbone for centernet, one of the ['mobilenet', 'vgg']", default='vgg')

    args = parser.parse_args()
    args.batch_size = 1

    net = load_model(args)
    dataset, loader = load_data(args, val_only=True)
    outdir = args.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    i = 0
    for im, labels, im_path in tqdm(loader):
        im_path = im_path[0]
        i += 1
        try:
            pred = detect(net, im)
            bboxes, classes = decode(pred)
            im = visualize(im_path, bboxes, classes)
            cv2.imwrite(os.path.join(outdir, '{}.jpg'.format(i)), im)
        except:
            pass
