import torch
import torch.nn as nn


class PointLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, gt):
        mask = gt.gt(0)
        pred_ = pred[mask]
        gt_ = pred[gt]
        loss = self.loss(pred_, gt_)
        loss = loss / (mask.float().sum() + 1e-4)
        return loss


class RegLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, gt):
        mask = gt.gt(0)
        pred_ = pred[mask]
        gt_ = gt[mask]
        loss = self.loss(pred_, gt_)
        loss = loss / (mask.float().sum() + 1e-4)
        return loss
