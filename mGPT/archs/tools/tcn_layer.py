import torch.nn.functional  as F
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import math

class TemporalBlock(nn.Module):
    """
    Bidirectional building block for TCN.
    Uses symmetric padding so each position sees both past and future context.
    Requires odd kernel_size (e.g. 3) for exact length preservation.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0):
        super(TemporalBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                self.conv2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Bidirectional Temporal Convolutional Network with exponentially
    increasing dilation rates for multi-scale temporal modeling.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0, dilation_growth_rate=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = dilation_growth_rate ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class nonlinearity(nn.Module):
    """
    Custom nonlinearity module implementing Swish activation function.
    Swish(x) = x * sigmoid(x)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Apply Swish activation: x * sigmoid(x)"""
        return x * torch.sigmoid(x)

class DownsamplingTCN(nn.Module):
    """
    Downsampling module: strided convolution followed by bidirectional TCN.
    """
    def __init__(self, num_inputs, num_outputs, tcn_layers=2, kernel_size=3, stride=2,
                 dropout=0, dilation_growth_rate=2):
        super(DownsamplingTCN, self).__init__()

        self.stride = stride
        padding = (kernel_size - 1) // 2

        self.downsample_conv = nn.Conv1d(num_inputs, num_outputs, kernel_size,
                                       stride=stride, padding=padding)

        tcn_channels = [num_outputs] * tcn_layers
        self.tcn = TemporalConvNet(num_outputs, tcn_channels, dropout=dropout,
                                   dilation_growth_rate=dilation_growth_rate)

    def forward(self, x):
        x = self.downsample_conv(x)
        x = self.tcn(x)
        return x

class UpsamplingTCN(nn.Module):
    """
    Upsampling module: nearest-neighbor upsampling followed by bidirectional TCN.
    """
    def __init__(self, num_inputs, num_outputs, tcn_layers=2, kernel_size=3, scale_factor=2,
                 dropout=0, dilation_growth_rate=2):
        super(UpsamplingTCN, self).__init__()

        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')

        if tcn_layers > 1:
            tcn_channels = [num_inputs] * (tcn_layers - 1) + [num_outputs]
        else:
            tcn_channels = [num_outputs]

        self.tcn = TemporalConvNet(num_inputs, tcn_channels, dropout=dropout,
                                   dilation_growth_rate=dilation_growth_rate)

    def forward(self, x):
        x = self.upsample(x)
        x = self.tcn(x)
        return x

class Temporal_Encoder(nn.Module):
    """
    Temporal Encoder using bidirectional TCN for hierarchical temporal
    feature extraction with progressive downsampling.

    Each downsampling level: StridedConv(↓) → BidirectionalTCN(depth layers).
    The bidirectional TCN replaces the original causal-TCN + Resnet1D stack,
    eliminating redundancy while allowing each position to access both
    past and future context.
    """
    def __init__(self, input_emb_width=361, output_emb_width=512, down_t=3, stride_t=2,
                 width=512, depth=3, dilation_growth_rate=3, dropout=0.1,
                 code_num=None, code_dim=None, **kwargs):
        super().__init__()
        self.down_t = down_t
        self.stride_t = stride_t

        self.initial_proj = nn.Sequential(
            nn.Conv1d(input_emb_width, width, 3, padding=1),
            nn.ReLU()
        )

        self.downsample_layers = nn.ModuleList()
        for i in range(down_t):
            block = DownsamplingTCN(
                num_inputs=width,
                num_outputs=width,
                tcn_layers=depth,
                stride=stride_t,
                dropout=dropout,
                dilation_growth_rate=dilation_growth_rate
            )
            self.downsample_layers.append(block)

        self.final_proj = nn.Conv1d(width, output_emb_width, 3, padding=1)

    def forward(self, x):
        x = self.initial_proj(x)
        for downsample_layer in self.downsample_layers:
            x = downsample_layer(x)
        x = self.final_proj(x)
        return x

class Temporal_Decoder(nn.Module):
    """
    Temporal Decoder using bidirectional TCN for progressive upsampling
    and temporal feature reconstruction.

    Each upsampling level: NearestUpsample(↑) → BidirectionalTCN(depth layers).
    Symmetric to Temporal_Encoder.
    """
    def __init__(self, input_emb_width=512, output_emb_width=361, down_t=3, stride_t=2,
                 width=512, depth=3, dilation_growth_rate=3, dropout=0.1,
                 code_num=None, code_dim=None, **kwargs):
        super().__init__()
        self.down_t = down_t
        self.stride_t = stride_t

        self.initial_proj = nn.Sequential(
            nn.Conv1d(output_emb_width, width, 3, padding=1),
            nn.ReLU()
        )

        self.upsample_layers = nn.ModuleList()
        for i in range(down_t):
            block = UpsamplingTCN(
                num_inputs=width,
                num_outputs=width,
                tcn_layers=depth,
                scale_factor=stride_t,
                dropout=dropout,
                dilation_growth_rate=dilation_growth_rate
            )
            self.upsample_layers.append(block)

        self.final_block = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, padding=1)
        )

    def forward(self, x):
        x = self.initial_proj(x)
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
        x = self.final_block(x)
        return x
