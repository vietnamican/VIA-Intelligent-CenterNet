import torch
import torch.nn as nn
from torchmetrics import Metric


class PointLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, gt):
        mask = gt.gt(0)
        pred = pred[mask]
        gt = gt[mask]
        loss = self.loss(pred, gt)
        loss = loss / (mask.float().sum() + 1e-4)
        return loss


class RegLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, gt):
        mask = gt.gt(0)
        pred = pred[mask]
        gt = gt[mask]
        loss = self.loss(pred, gt)
        loss = loss / (mask.float().sum() + 1e-4)
        return loss


class AverageMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, correct, total):
        # print(correct)
        # print(total)
        self.correct += correct
        self.total += total

    def compute(self):
        return self.correct.float() / self.total
