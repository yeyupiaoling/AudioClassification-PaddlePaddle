import paddle.nn as nn
import paddle.nn.functional as F

from ppacls.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling
from ppacls.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling
from ppacls.models.utils import BatchNorm1d


class TDNN(nn.Layer):
    def __init__(self, num_class, input_size=80, channels=512, embd_dim=192, pooling_type="ASP"):
        super(TDNN, self).__init__()
        self.embd_dim = embd_dim
        self.td_layer1 = nn.Conv1D(in_channels=input_size, out_channels=channels, dilation=1, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1D(channels)
        self.td_layer2 = nn.Conv1D(in_channels=channels, out_channels=channels, dilation=2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1D(channels)
        self.td_layer3 = nn.Conv1D(in_channels=channels, out_channels=channels, dilation=3, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1D(channels)
        self.td_layer4 = nn.Conv1D(in_channels=channels, out_channels=channels, dilation=1, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1D(channels)
        self.td_layer5 = nn.Conv1D(in_channels=channels, out_channels=channels, dilation=1, kernel_size=1, stride=1)

        if pooling_type == "ASP":
            self.pooling = AttentiveStatisticsPooling(channels, attention_channels=128)
            self.bn5 = BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(channels, 128)
            self.bn5 = BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn5 = BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn5 = BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')
        # 分类层
        self.output = nn.Linear(self.embd_dim, num_class)

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (paddle.Tensor): Input data with shape (N, time, freq).

        Returns:
            paddle.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.transpose([0, 2, 1])
        x = F.relu(self.td_layer1(x))
        x = self.bn1(x)
        x = F.relu(self.td_layer2(x))
        x = self.bn2(x)
        x = F.relu(self.td_layer3(x))
        x = self.bn3(x)
        x = F.relu(self.td_layer4(x))
        x = self.bn4(x)
        x = F.relu(self.td_layer5(x))
        out = self.bn5(self.pooling(x))
        out = self.bn6(self.linear(out))
        out = self.output(out)
        return out
