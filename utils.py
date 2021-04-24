import os

import numpy as np
import cv2
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

cls = ['stop', 'left', 'right', 'straight', 'no_left', 'no_right']


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
                                 pin_memory=True, num_workers=num_workers, shuffle=True)

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


def iou(box1, box2):
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[2]
    y21 = box1[3]

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[2]
    y22 = box2[3]

    xx1 = max(x11, x12)
    yy1 = max(y11, y12)
    xx2 = min(x21, x22)
    yy2 = min(y21, y22)

    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)

    overlap = w * h
    overall = (x21 - x11 + 1) * (y21 - y11 + 1) + \
        (x22 - x12 + 1) * (y22 - y12 + 1) - overlap
    return overlap / 1.0 / overall


def nms(boxes, scores, nms_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=bool)

    keep = []
    for _i in range(num_detections):
        i = order[_i]
        if suppressed[i]:
            continue
        keep.append(i)

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_detections):
            j = order[_j]
            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh or inter / iarea >= nms_thresh or inter / areas[j] >= nms_thresh:
                suppressed[j] = True

    return keep


def detect(net, im):
    with torch.no_grad():
        out = net(im)
    return out[0]


def decode(out):
    hm = out['hm']
    wh = out['wh']
    hm.squeeze_()
    wh.squeeze_()

    hm = hm.numpy()
    hm[hm < cfg.threshold] = 0
    cls, ys, xs = np.nonzero(hm)
    bboxes = []
    scores = []
    classes = []
    for cl, y, x in zip(cls, ys, xs):
        w = wh[0][y, x]
        h = wh[1][y, x]
        width = np.exp(w)
        height = np.exp(h)

        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2
        bboxes.append([left, top, right, bottom])
        scores.append(hm[cl, y, x])
        classes.append(cl)

    bboxes = np.array(bboxes)   
    scores = np.array(scores)
    classes = np.array(classes)
    if len(bboxes) == 0:
        print('no_bbox')
        return bboxes
    keep_indexes = nms(bboxes, scores, 0.4)
    print(classes[keep_indexes])
    return bboxes[keep_indexes], classes[keep_indexes]


def visualize(im_path, bboxes, classes):
    im = cv2.imread(im_path)
    im = cv2.resize(im, (240, 176))
    for bbox, cl in zip(bboxes, classes):
        left, top, right, bottom = bbox
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        cv2.rectangle(im, (left, top), (right, bottom), (255, 0, 0), 2)
        # cv2.putText(im, cls[cl], (left, top), fontFace=cv2.FONT_HERSHEY_SIMPLEX, bottomLeftCornerOfText=(
        #     10, 500), fontScale=1, fontColor=(255, 255, 255), lineType=2)
        cv2.putText(im, cls[cl], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 0), 1, cv2.LINE_AA)

    return im
