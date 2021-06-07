from torch import nn
import torch.utils.model_zoo as model_zoo

from ..base import Base, VGGBlock
from ..layers import ConvBNReLU, InvertedResidual

__all__ = ['MobileNetV2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class MobileNetV2(Base):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.channels = [32, 24, 32, 96, 320]
        self.first_conv = nn.Sequential(
            ConvBNReLU(3, 32, stride=1),
        )
        self.feature_1 = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),
        )
        self.feature_2 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
        )
        self.feature_4 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
        )
        self.feature_6 = nn.Sequential(
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 320, 1, 6),
        )

    def forward(self, x):
        y = []
        x = self.feature_2(self.feature_1(self.first_conv(x)))
        y.append(x)
        x = self.feature_4(x)
        y.append(x)
        y.append(self.feature_6(x))
        return y

def load_mobile_net(pretrained=True):
    model = MobileNetV2()
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mobilenet_v2'],
                                              progress=True)
        model.migrate(state_dict, force=True, verbose=1)
    return model