"""Causal encoder/decoder modules for the streaming MambAttention model.

Key changes relative to the non-causal MambAttention codec:

1. ``ChannelLayerNorm`` replaces ``InstanceNorm2d`` — normalization only
   over the channel axis (no cross-time/cross-frequency statistics), which
   keeps the network causal.
2. All time-axis convolutions use *causal* padding (all padding on the left),
   via ``get_causal_padding_2d``.
3. Upsampling uses sub-pixel convolution (``SPConvTranspose2d``) instead of
   ``nn.ConvTranspose2d`` — checkerboard-free and streaming-friendly.
"""

import torch
import torch.nn as nn
from einops import rearrange


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis only.

    Input shape: [B, C, T, F]. Normalization statistics are computed per
    (batch, time, freq) position over C, so no cross-time or cross-frequency
    information is used -> causal-safe.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # [B, C, T, F] -> [B, T, F, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # [B, T, F, C] -> [B, C, T, F]
        x = x.permute(0, 3, 1, 2)
        return x


def get_causal_padding_2d(kernel_size, dilation=(1, 1)):
    """Causal padding along the time axis, symmetric along frequency.

    kernel_size: (kt, kf); dilation: (dt, df). Returns
    (pad_left_f, pad_right_f, pad_top_t, pad_bottom_t) for ``F.pad`` /
    ``ConstantPad2d`` (left, right, top, bottom).
    """
    pad_t = kernel_size[0] * dilation[0] - dilation[0]  # all on the left (top)
    pad_f = (kernel_size[1] * dilation[1] - dilation[1]) // 2
    return (pad_f, pad_f, pad_t, 0)


class SPConvTranspose2d(nn.Module):
    """Sub-pixel convolution implementing 2D upsampling.

    A strided (transposed) upsampling is expressed as a regular ``Conv2d``
    producing ``out_channels * r`` channels followed by a periodic-shuffle
    reshape. ``padding`` is a 4-tuple passed to ``ConstantPad2d`` (applied
    before the convolution), so the operation is explicit and causal-aware.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad = nn.ConstantPad2d(padding, value=0.)
        self.out_channels = out_channels
        self.r = r
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, padding=0
        )

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    """Dense block with causal, dilated convolutions."""

    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.cfg = cfg
        self.depth = depth
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.dense_block = nn.ModuleList()

        for i in range(depth):
            dil = 2 ** i
            pad = get_causal_padding_2d(kernel_size, (dil, 1))
            dense_conv = nn.Sequential(
                nn.ConstantPad2d(pad, 0.0),
                nn.Conv2d(
                    self.hid_feature * (i + 1), self.hid_feature, kernel_size,
                    dilation=(dil, 1), padding=0,
                ),
                ChannelLayerNorm(self.hid_feature),
                nn.PReLU(self.hid_feature),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    """Dense encoder with causal convolutions and frequency downsampling."""

    def __init__(self, cfg):
        super(DenseEncoder, self).__init__()
        self.cfg = cfg
        self.input_channel = cfg['model_cfg']['input_channel']
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (1, 1)),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature),
        )

        self.dense_block = DenseBlock(cfg, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature),
        )

    def forward(self, x):
        # x: [B, C, T, F]
        x = self.dense_conv_1(x)  # [B, hid_feature, T, F]
        x = self.dense_block(x)  # [B, hid_feature, T, F]
        x = self.dense_conv_2(x)  # [B, hid_feature, T, F//2]
        return x


class MagDecoder(nn.Module):
    """Magnitude decoder (outputs a mask in [0, beta])."""

    def __init__(self, cfg):
        super(MagDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.beta = cfg['model_cfg']['beta']

        # Upsample frequency by 2 via sub-pixel convolution (causal padding).
        self.up_conv1 = nn.Sequential(
            SPConvTranspose2d(
                self.hid_feature, self.hid_feature, (1, 3),
                get_causal_padding_2d((1, 3)), 2,
            ),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature),
        )

        self.mask_conv = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
        )

    def forward(self, x):
        # x: [B, C, T, F//2]
        x = self.dense_block(x)
        x = self.up_conv1(x)  # [B, C, T, F]
        x = self.mask_conv(x)  # [B, 1, T, F]
        # sigmoid mask in [0, beta]
        x = self.beta * torch.sigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    """Phase decoder (outputs phase via atan2 of real/imag)."""

    def __init__(self, cfg):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']

        self.up_conv1 = nn.Sequential(
            SPConvTranspose2d(
                self.hid_feature, self.hid_feature, (1, 3),
                get_causal_padding_2d((1, 3)), 2,
            ),
            ChannelLayerNorm(self.hid_feature),
            nn.PReLU(self.hid_feature),
        )

        self.phase_conv_r = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))

    def forward(self, x):
        # x: [B, C, T, F//2]
        x = self.dense_block(x)
        x = self.up_conv1(x)  # [B, C, T, F]
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x
