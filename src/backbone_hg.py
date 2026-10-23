"""Single RODNet HG feature backbone, derived from upstream MIT code.

Source: yizhou-wang/RODNet, commit 44fd61fa7b433b920469bd8455a54d5676e57fd8.
Copyright (c) 2021 Yizhou Wang. See third_party/hg/hg.LICENSE.
RODEncode/RODDecode are unchanged. Only input/output adaptation and removal of
the category confidence/feedback heads are RadarVLA-specific.
"""
from torch import nn
from .backbone_adapter import TemporalFeatureAdapter

class RODEncode(nn.Module):

    def __init__(self):
        super(RODEncode, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))

        self.skipconv1a = nn.Conv3d(in_channels=64, out_channels=64,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.skipconv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                    kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.skipconv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.skipconv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                    kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.skipconv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.skipconv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                    kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)

        self.skipbn1a = nn.BatchNorm3d(num_features=64)
        self.skipbn1b = nn.BatchNorm3d(num_features=64)
        self.skipbn2a = nn.BatchNorm3d(num_features=128)
        self.skipbn2b = nn.BatchNorm3d(num_features=128)
        self.skipbn3a = nn.BatchNorm3d(num_features=256)
        self.skipbn3b = nn.BatchNorm3d(num_features=256)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.skipbn1a(self.skipconv1a(x)))
        x1 = self.relu(self.skipbn1b(self.skipconv1b(x1)))
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)

        x2 = self.relu(self.skipbn2a(self.skipconv2a(x)))
        x2 = self.relu(self.skipbn2b(self.skipconv2b(x2)))
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)

        x3 = self.relu(self.skipbn3a(self.skipconv3a(x)))
        x3 = self.relu(self.skipbn3b(self.skipconv3b(x3)))
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/4, 16, 16)

        return x, x1, x2, x3


class RODDecode(nn.Module):

    def __init__(self):
        super(RODDecode, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        # self.upsample = nn.Upsample(size=(rodnet_configs['win_size'], radar_configs['ramap_rsize'],
        #                                   radar_configs['ramap_asize']), mode='nearest')

    def forward(self, x, x1, x2, x3):
        x = self.prelu(self.convt1(x + x3))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.convt2(x + x2))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.convt3(x + x1)  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        return x


class RODNetHGFeatures(TemporalFeatureAdapter):
    def __init__(self, hidden_dim=64):
        super().__init__(hidden_dim)
        self.conv1a = nn.Conv3d(2, 32, (9, 5, 5), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(32, 64, (9, 5, 5), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(32)
        self.bn1b = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.encode = RODEncode()
        self.decode = RODDecode()

    def features(self, x, validity=None):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        return self.decode(*self.encode(x))
