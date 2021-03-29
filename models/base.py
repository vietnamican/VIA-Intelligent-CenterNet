from typing import Dict, Iterable, List, Optional, Union
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl


class Base(pl.LightningModule):
    def __init__(self):
        super(Base, self).__init__()

    def remove_num_batches_tracked(self, state_dict):
        new_state_dict = {}
        for name, p in state_dict.items():
            if not 'num_batches_tracked' in name:
                new_state_dict[name] = p
        return new_state_dict

    def migrate(
            self,
            state_dict: Dict,
            other_state_dict=None,
            force=False,
            verbose=2
    ):
        '''
        verbose=0: do not print
        verbose=1: print status of migrate: all is migrated or something
        verbose=2: print all of modules had been migrated
        '''
        if verbose == 0:
            def status(i, string):
                pass

            def conclude(is_all_migrated):
                pass
        elif verbose == 1:
            def status(i, string):
                pass

            def conclude(is_all_migrated):
                if is_all_migrated:
                    print("all modules had been migrated")
                else:
                    print("Some modules hadn't been migrated")
        elif verbose == 2:
            def status(i, string):
                print(i, string)

            def conclude(is_all_migrated):
                if is_all_migrated:
                    print("all modules had been migrated")
                else:
                    print("Some modules hadn't been migrated")

        if other_state_dict is None:
            des_state_dict = self.state_dict()
            source_state_dict = state_dict
        else:
            des_state_dict = state_dict
            source_state_dict = other_state_dict

        des_state_dict = self.remove_num_batches_tracked(des_state_dict)
        source_state_dict = self.remove_num_batches_tracked(source_state_dict)
        is_all_migrated = True

        if not force:
            state_dict_keys = source_state_dict.keys()
            with torch.no_grad():
                for i, (name, p) in enumerate(des_state_dict.items()):
                    if name in state_dict_keys:
                        _p = source_state_dict[name]
                        if p.data.shape == _p.shape:
                            status(i, name)
                            p.copy_(_p)
                        else:
                            is_all_migrated = False
                    else:
                        is_all_migrated = False

        else:
            print('Force migrating...')
            with torch.no_grad():
                for i, ((name, p), (_name, _p)) in enumerate(zip(des_state_dict.items(), source_state_dict.items())):
                    if p.shape == _p.shape:
                        status(i, 'copy to {} from {}'.format(name, _name))
                        p.copy_(_p)
                    else:
                        is_all_migrated = False
        conclude(is_all_migrated)


class ConvBatchNormRelu6(Base):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if 'with_relu' not in kwargs:
            self.with_relu = True
        else:
            self.with_relu = kwargs['with_relu']
            kwargs.pop('with_relu', None)
        if 'with_bn' not in kwargs:
            self.with_bn = True
        else:
            self.with_bn = kwargs['with_bn']
            kwargs.pop('with_bn', None)

        self.args = args
        self.kwargs = kwargs

        self.cbr = nn.Sequential()
        self.cbr.add_module('conv', nn.Conv2d(*args, **kwargs))
        if self.with_bn:
            outplanes = args[1]
            self.cbr.add_module('bn', nn.BatchNorm2d(int(outplanes)))
        if self.with_relu:
            self.cbr.add_module('relu', nn.ReLU6(inplace=True))

    def forward(self, x):
        return self.cbr(x)


class VGGBlock(Base):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=False,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvBatchNormRelu6(
            inplanes, planes, kernel_size=3, padding=1, bias=False, stride=stride, with_relu=False)
        self.identity_layer = ConvBatchNormRelu6(
            inplanes, planes, kernel_size=1, padding=0, bias=False, stride=stride, with_relu=False)
        self.relu = nn.ReLU6(inplace=True)
        self.stride = stride
        if inplanes == planes and stride == 1:
            self.skip_layer = nn.BatchNorm2d(num_features=inplanes)
        else:
            self.skip_layer = None
        if self.skip_layer is not None:

            def _forward(self, x):
                conv3 = self.conv1(x)
                identity = self.identity_layer(x)
                skip = self.skip_layer(x)
                return self.relu(conv3 + identity + skip)
        else:

            def _forward(self, x):
                conv3 = self.conv1(x)
                identity = self.identity_layer(x)
                return self.relu(conv3 + identity)
        self._forward = partial(_forward, self)

    def forward(self, x):
        return self._forward(x)
