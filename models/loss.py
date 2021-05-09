import torch
import torch.nn as nn


class PointLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, gt):
        mask = gt.gt(0).float()
        loss = self.loss(pred * mask, gt)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, gt):
        mask = gt.gt(0).float()
        loss = self.loss(pred * mask, gt)
        loss = loss / (mask.sum() + 1e-4)
        return loss
