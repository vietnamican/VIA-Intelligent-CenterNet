import os
from tqdm import tqdm
import argparse

import cv2
import torch
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms as T
import mathplotlib.pyplot as plt

# local imports
from config import Config as cfg
from models.backbone.vgg import VGG
from models.model import Model
from utils import load_data, load_model as _load_model, detect, decode, visualize


transformer = T.Compose([
    T.ToPILImage(),
    T.Resize((176, 240)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


pl.seed_everything(42)


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
    parser.add_argument("--image-path", type=str,
                        help='Path to validation image folder', default='via-trafficsign/images/val')
    parser.add_argument(
        "--device", type=str, help="Choose what device to train, one of the: ['cpu', 'gpu', 'tpu'], tpu is unavailable now", default='cpu')
    parser.add_argument(
        '--out-path', type=str, help='Choose logdir for tensorboard logger', default='traffic_logs/training')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint to load from', required=True)
    parser.add_argument(
        '--backbone', type=str, help="Choose backbone for centernet, one of the ['mobilenet', 'vgg']", default='vgg')

    args = parser.parse_args()

    net = load_model(args)
    im_path = args.image_path
    im = cv2.imread(im_path)
    im = transformer(im)
    try:
        pred = detect(net, im)
        bboxes, classes = decode(pred)
        im = visualize(im_path, bboxes, classes)
        plt.imshow(im[:,:,::-1])
        plt.show()
    except:
        pass
