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
from utils import load_data, load_model as _load_model


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
    ys, xs = np.nonzero(hm)
    bboxes = []
    scores = []
    for y, x in zip(ys, xs):
        w = wh[0][y, x]
        h = wh[1][y, x]
        width = np.exp(w)
        height = np.exp(h)

        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2
        bboxes.append([left, top, right, bottom])
        scores.append(hm[y, x])

    bboxes = np.array(bboxes)
    if len(bboxes) == 0:
        return bboxes
    keep_indexes = nms(bboxes, scores, 0.4)
    return bboxes[keep_indexes]


def visualize(im_path, bboxes):
    im = cv2.imread(im_path)
    im = cv2.resize(im, (240, 176))
    for bbox in bboxes:
        left, top, right, bottom = bbox
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        cv2.rectangle(im, (left, top), (right, bottom), (255, 0, 0), 2)
    return im


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
        try:
            i += 1
            pred = detect(net, im)
            bboxes = decode(pred)
            im = visualize(im_path, bboxes)
            cv2.imwrite(os.path.join(outdir, '{}.jpg'.format(i)), im)
        except:
            pass
