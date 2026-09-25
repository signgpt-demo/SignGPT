import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import math


class Chomp1d(nn.Module):
    """Remove trailing elements for causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class CausalTemporalBlock(nn.Module):
    """
    Causal building block for TCN.
    Uses left-only padding so each position only depends on current and past inputs.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0):
        super().__init__()
        padding = dilation * (kernel_size - 1)

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)

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


class CausalTemporalConvNet(nn.Module):
    """Causal TCN with exponentially increasing dilation rates."""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0, dilation_growth_rate=2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = dilation_growth_rate ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [CausalTemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                          dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalDownsamplingTCN(nn.Module):
    """Downsampling with strided causal convolution + causal TCN."""
    def __init__(self, num_inputs, num_outputs, tcn_layers=2, kernel_size=3, stride=2,
                 dropout=0, dilation_growth_rate=2):
        super().__init__()

        self.stride = stride
        padding = (kernel_size - 1) * stride

        self.downsample_conv = nn.Conv1d(num_inputs, num_outputs, kernel_size,
                                       stride=stride, padding=padding)

        tcn_channels = [num_outputs] * tcn_layers
        self.tcn = CausalTemporalConvNet(num_outputs, tcn_channels, dropout=dropout,
                                        dilation_growth_rate=dilation_growth_rate)
        self.chomp = Chomp1d(padding)

    def forward(self, x):
        x = self.downsample_conv(x)
        x = self.chomp(x)
        x = self.tcn(x)
        return x


class CausalUpsamplingTCN(nn.Module):
    """Upsampling with nearest-neighbor + causal TCN.
    The decoder's outer forward pass crops/pads the temporal dim to the
    encoder's input length, so this block does not chomp on its own.
    The non-symmetric chomp would make the upsample produce 0-length
    tensors for some input sizes."""
    def __init__(self, num_inputs, num_outputs, tcn_layers=2, kernel_size=3, scale_factor=2,
                 dropout=0, dilation_growth_rate=2):
        super().__init__()

        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')

        if tcn_layers > 1:
            tcn_channels = [num_inputs] * (tcn_layers - 1) + [num_outputs]
        else:
            tcn_channels = [num_outputs]

        self.tcn = CausalTemporalConvNet(num_inputs, tcn_channels, dropout=dropout,
                                        dilation_growth_rate=dilation_growth_rate)

    def forward(self, x):
        x = self.upsample(x)
        x = self.tcn(x)
        return x


class CausalTemporal_Encoder(nn.Module):
    """
    Temporal Encoder using causal TCN for autoregressive-style feature extraction
    with progressive downsampling. Each position only depends on current and past inputs.
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
            block = CausalDownsamplingTCN(
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


class CausalTemporal_Decoder(nn.Module):
    """
    Temporal Decoder using causal TCN for progressive upsampling.
    Symmetric to CausalTemporal_Encoder.
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
            block = CausalUpsamplingTCN(
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
