import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import numpy as np

from .centernet import CenterNet
from .base import Base
from .loss import PointLoss, RegLoss


class Model(CenterNet):
    def __init__(self, base):
        super().__init__(base, {'hm': 1, 'wh': 2})
        self.threshold = 0.4
        self.heatmap_loss = PointLoss()
        self.wh_loss = RegLoss()

    def training_step(self, batch, batch_idx):
        data, labels, *_ = batch
        out = self(data)
        heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
        l_heatmap = self.heatmap_loss(heatmaps, labels[:, 0])
        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = self.wh_loss(whs, labels[:, [1, 2]])

        self.log_dict({'t_heat': l_heatmap,
                       't_size': l_wh}, prog_bar=True)
        loss = l_heatmap + l_wh * 0.1
        self.log_dict({'train_loss': loss})

        return loss

    def validation_step(self, batch, batch_idx):
        data, labels, *_ = batch
        out = self(data)
        heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
        l_heatmap = self.heatmap_loss(heatmaps, labels[:, 0])
        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = self.wh_loss(whs, labels[:, [1, 2]])

        self.log_dict({'v_heat': l_heatmap,
                       'v_size': l_wh}, prog_bar=False)
        loss = l_heatmap + l_wh * 0.1
        self.log_dict({'val_loss': loss})

        return loss

    def test_step(self, batch, batch_idx):
        data, labels, *_ = batch
        out = self(data)
        heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
        l_heatmap = self.heatmap_loss(heatmaps, labels[:, 0])
        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = self.wh_loss(whs, labels[:, [1, 2]])
        self.log_dict({'heat': l_heatmap,
                       'size': l_wh}, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0005, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def release(self):
        is_self = True
        for module in self.modules():
            if is_self:
                is_self = False
                continue
            if hasattr(module, 'release'):
                module.release()
