from torch import nn

from ..base import Base, VGGBlock, ConvBatchNormRelu6

class VGG(Base):
    def __init__(self):
        super().__init__()
        self.channels = [32, 32, 64, 128, 256]
        self.conv1 = ConvBatchNormRelu6(3, 32, 3, padding=1, bias=False)
        self.feature_1 = nn.Sequential(
            VGGBlock(32, 32, 2),
            VGGBlock(32, 32, 1),
            VGGBlock(32, 32, 1),
        )
        self.feature_2 = nn.Sequential(
            VGGBlock(32, 64, 2),
            VGGBlock(64, 64, 1),
            VGGBlock(64, 64, 1),
        )
        self.feature_3 = nn.Sequential(
            VGGBlock(64, 128, 2),
            VGGBlock(128, 128, 1),
            VGGBlock(128, 128, 1),
            VGGBlock(128, 128, 1),
        )
        self.feature_4 = nn.Sequential(
            VGGBlock(128, 256, 2),
            VGGBlock(256, 256, 1),
            VGGBlock(256, 256, 1),
        )

    def forward(self, x):
        y = []
        x = self.conv1(x)
        y.append(x)
        x = self.feature_1(x)
        y.append(x)
        x = self.feature_2(x)
        y.append(x)
        x = self.feature_3(x)
        y.append(x)
        y.append(self.feature_4(x))
        return y

    def release(self):
        is_self = True
        for module in self.modules():
            if is_self:
                is_self = False
                continue
            if hasattr(module, 'release'):
                module.release()