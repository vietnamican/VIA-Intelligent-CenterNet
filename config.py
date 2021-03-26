import torch
from torchvision import transforms as T


class Config:
    # preprocess
    insize = [416, 416]
    channels = 3
    downscale = 4
    sigma = 2.65

    # training
    epoch = 90
    lr = 5e-4
    batch_size = 32
    pin_memory = True
    num_workers = 18
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # inference
    threshold = 0.5