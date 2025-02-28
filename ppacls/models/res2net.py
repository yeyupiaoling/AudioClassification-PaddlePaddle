import math

import paddle
import paddle.nn as nn

from ppacls.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling
from ppacls.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling
from ppacls.models.utils import BatchNorm1d


class Bottle2neck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2D(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm2D(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2D(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2D(width, width, kernel_size=3, stride=stride, padding=1))
            bns.append(nn.BatchNorm2D(width))
        self.convs = nn.LayerList(convs)
        self.bns = nn.LayerList(bns)

        self.conv3 = nn.Conv2D(width * scale, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = paddle.split(out, self.scale, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = paddle.concat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = paddle.concat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = paddle.concat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Layer):

    def __init__(self, num_class, input_size, m_channels=32, layers=[3, 4, 6, 3], base_width=32, scale=2, embd_dim=192,
                 pooling_type="ASP"):
        super(Res2Net, self).__init__()
        self.inplanes = m_channels
        self.base_width = base_width
        self.scale = scale
        self.embd_dim = embd_dim
        self.conv1 = nn.Conv2D(1, m_channels, kernel_size=7, stride=3, padding=1)
        self.bn1 = nn.BatchNorm2D(m_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottle2neck, m_channels, layers[0])
        self.layer2 = self._make_layer(Bottle2neck, m_channels*2, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottle2neck, m_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottle2neck, m_channels * 8, layers[3], stride=2)

        if input_size < 96:
            cat_channels = m_channels * 8 * Bottle2neck.expansion * (input_size // self.base_width)
        else:
            cat_channels = m_channels * 8 * Bottle2neck.expansion * (
                        input_size // self.base_width - int(math.sqrt(input_size / 64)))
        if pooling_type == "ASP":
            self.pooling = AttentiveStatisticsPooling(cat_channels, attention_channels=128)
            self.bn2 = BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(cat_channels, 128)
            self.bn2 = BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn2 = BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn2 = BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

        self.output = nn.Linear(self.embd_dim, num_class)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.base_width, scale=self.scale)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.base_width, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose([0, 2, 1])
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape([x.shape[0], -1, x.shape[-1]])
        x = self.pooling(x)
        x = self.bn2(x)
        x = self.linear(x)
        x = self.bn3(x)
        out = self.output(x)
        return out
