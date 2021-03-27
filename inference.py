import os
import os.path as osp
from tqdm import tqdm

import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# local imports
from config import Config as cfg
from models.backbone.vgg import VGG
from models.model import Model
# from utils import VisionKit
from datasets import TraficDataset
from models.loss import AverageMetric

traindataset = TraficDataset('via-trafficsign/images/train', 'via-trafficsign/labels/train', 'train')
trainloader = DataLoader(traindataset, batch_size=1,
                        pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)

valdataset = TraficDataset('via-trafficsign/images/val', 'via-trafficsign/labels/val', 'val')
valloader = DataLoader(valdataset, batch_size=1,
                        pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
device = 'cpu'
precision = AverageMetric()
recall = AverageMetric()

def load_model():
    net = Model(VGG)
    checkpoint_path = 'checkpoint-epoch=51-val_loss=0.0034.ckpt'
    if device == 'cpu':
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    net.eval()
    net.migrate(state_dict, force=True, verbose=2)
    net.eval()
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
    suppressed = np.zeros((num_detections,), dtype=np.bool)

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
            if ovr >= nms_thresh:
                suppressed[j] = True

    return keep

transformer = {
    'train': T.Compose([
        T.ToPILImage(),
        T.ColorJitter(0.5, 0.5, 0.5, 0.5),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'val': T.Compose([
        T.ToPILImage(),
        T.Resize((176, 240)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

def detect(net, im):
    # data = transformer['val'](im)
    # data = data[None, ...]
    with torch.no_grad():
        out = net(im)
    return out[0]


def decode(out):
    hm = out['hm']
    wh = out['wh']
    # hm = VisionKit.nms(hm, kernel=3)
    hm.squeeze_()
    wh.squeeze_()

    hm = hm.numpy()
    hm[hm < cfg.threshold] = 0
    # for line in hm:
    #     print(line)
    # print(hm)
    ys, xs = np.nonzero(hm)
    bboxes = []
    scores = []
    landmarks = []
    for y, x in zip(ys, xs):
        # print(x, y)
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

        # landmark
    bboxes = np.array(bboxes)
    if len(bboxes) == 0:
        return bboxes
    keep_indexes = nms(bboxes, scores, 0.4)
    return bboxes[keep_indexes]


def visualize(im, bboxes):
    pass

# def manhattan_distance(box1, box2):
#     center1 = (box1[2] - box1[0], box1[3] - box1[1])
#     center2 = (box2[2] - box2[0], box2[3] - box2[1])
#     return abs(center1[0] - center2[0]) + abs(center1[1] - center2[1])

def calculate_metrics(pred_bboxes, gt_bboxes):
    # print(len(pred_bboxes))
    iou_threshold = 0.4
    is_occupied = [False]*gt_bboxes.shape[0]
    pred_index = []
    gt_index = []
    result = 0
    for i, pred_box in enumerate(pred_bboxes):
        max_iou = 0
        for j, gt_box in enumerate(gt_bboxes):

            if not is_occupied[j]:
                iou_distance = iou(pred_box, gt_box) 
                if iou_distance > max_iou:
                    closest_index = j
                    max_iou = iou_distance
        if max_iou > iou_threshold:
            result += 1
            is_occupied[closest_index] = True
            pred_index.append(i)
            gt_index.append(closest_index)
    return result, pred_index, gt_index

if not os.path.isdir('result189'):
    os.mkdir('result189')

if __name__ == '__main__':
    net = load_model()
    # i = 0
    for im, labels, bboxes in tqdm(trainloader):

        # try:
            # i += 1
            # if i == 10:
            #     break
        # print(labels[0])
        # for line in labels[0,0]:
        #     line[line < cfg.threshold] = 0
        #     print(line)
        x_ratio, y_ratio, width_ratio, height_ratio = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        x = np.round(x_ratio * valdataset.im_width)
        y = np.round(y_ratio * valdataset.im_height)
        width = np.round(width_ratio * valdataset.im_width)
        height = np.round(height_ratio * valdataset.im_height)
        left = x - width // 2
        right = x + width // 2
        top = y - height // 2
        bottom = y + height // 2
        gt_bboxes = np.concatenate([left, top, right, bottom], axis=-1)
        pred = detect(net, im)
        bboxes = decode(pred)
        loss = torch.nn.functional.mse_loss(pred['hm'], labels[0, 0], reduction='sum')
        print(loss)
        result, pred_index, gt_index = calculate_metrics(bboxes, gt_bboxes)
        # print(result)
        print(result, len(bboxes))
        print(result, len(gt_bboxes))
        precision.update(result, len(bboxes))
        recall.update(result, len(gt_bboxes))
        break
        # except:
        #     pass
    print(precision.compute())
    print(recall.compute())