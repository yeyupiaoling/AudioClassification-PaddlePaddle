import paddle.nn as nn

from ppacls.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling
from ppacls.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling
from ppacls.models.utils import BatchNorm1d


class SEBottleneck(nn.Layer):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.relu = nn.ReLU()
        self.se = SELayer(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape([b, c])
        y = self.fc(y).reshape([b, c, 1, 1])
        return x * y


class ResNetSE(nn.Layer):
    def __init__(self, num_class, input_size, layers=[3, 4, 6, 3], num_filters=[32, 64, 128, 256], embd_dim=192,
                 pooling_type="ASP"):
        super(ResNetSE, self).__init__()
        self.inplanes = num_filters[0]
        self.embd_dim = embd_dim
        self.conv1 = nn.Conv2D(1, num_filters[0], kernel_size=3, stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2D(num_filters[0])
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(SEBottleneck, num_filters[0], layers[0])
        self.layer2 = self._make_layer(SEBottleneck, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(SEBottleneck, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(SEBottleneck, num_filters[3], layers[3], stride=(2, 2))

        cat_channels = num_filters[3] * SEBottleneck.expansion * (input_size // 8)
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
        # 分类层
        self.output = nn.Linear(self.embd_dim, num_class)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose([0, 2, 1])
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

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
