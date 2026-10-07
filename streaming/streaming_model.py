"""Streaming wrapper for the causal MambAttention model.

This module reuses the weights of the non-causal ``MambAttention`` model and
wraps the temporal operations with explicit state caches so the model can run
in a 1-frame-in / 1-frame-out streaming fashion.

Three kinds of state are propagated across time:

1. ``StreamingCausalConv2d.cache`` — the past frames required by the temporal
   convolution (causal padding means we only need the ``(kt-1)*dt`` left frames).
2. ``StreamingCausalMambaBlock.conv_state`` — the internal buffer of the
   depth-wise 1D convolution preceding the SSM (over ``d_conv`` steps).
3. ``StreamingCausalMambaBlock.ssm_state`` — the recurrent hidden state of the
   selective state-space model.

The frequency dimension is non-streaming (fixed size per frame), so the
frequency Mamba and all frequency-axis convolutions are applied in full.

Reference: NVIDIA Realtime_REUSE ``online_inference.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.codec_module import get_causal_padding_2d


class StreamingCausalConv2d(nn.Module):
    """Wraps a causal Conv2d and caches past time frames.

    The wrapped conv must have ``padding=0`` (padding is handled by the cache +
    explicit frequency padding here).
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        kt, kf = conv.kernel_size
        dt, df = conv.dilation
        self.cache_len = (kt - 1) * dt
        self.register_buffer("cache", None, persistent=False)
        self.freq_pad = get_causal_padding_2d((1, kf), (dt, df))

    def reset(self):
        self.cache = None

    def forward(self, x, t_pad=None):
        # x: [B, C, 1, F]
        B, C, _, Fdim = x.shape

        if self.cache is None:
            if t_pad is None:
                t_pad = self.cache_len
            self.cache = torch.zeros(B, C, t_pad, Fdim, device=x.device, dtype=x.dtype)

        x_cat = torch.cat([self.cache, x], dim=2)  # [B, C, t_pad+1, F]
        y = self.conv(F.pad(x_cat, self.freq_pad, "constant", 0))
        self.cache = x_cat[:, :, -self.cache_len:, :]
        return y


class StreamingDenseBlock(nn.Module):
    """Streaming version of ``DenseBlock`` (caches time frames per conv)."""

    def __init__(self, dense_block):
        super().__init__()
        self.blocks = nn.ModuleList()
        for seq in dense_block.dense_block:
            # seq: [ConstantPad2d, Conv2d, ChannelLayerNorm, PReLU]
            conv = seq[1]
            self.blocks.append(nn.Sequential(
                StreamingCausalConv2d(conv),
                seq[2],  # ChannelLayerNorm
                seq[3],  # PReLU
            ))

    def reset(self):
        for b in self.blocks:
            b[0].reset()

    def forward(self, x):
        skip = x
        for b in self.blocks:
            y = b(skip)
            skip = torch.cat([y, skip], dim=1)
        return y


class StreamingDenseEncoder(nn.Module):
    """Streaming version of ``DenseEncoder``."""

    def __init__(self, encoder):
        super().__init__()
        # dense_conv_1 is (1,1) conv -> no temporal cache needed
        self.dense_conv_1 = encoder.dense_conv_1

        self.dense_block = StreamingDenseBlock(encoder.dense_block)

        # dense_conv_2 is (1,3) stride (1,2) on freq -> time kernel = 1, no cache
        self.dense_conv_2 = encoder.dense_conv_2

    def reset(self):
        self.dense_block.reset()

    def forward(self, x):
        # x: [B, C, 1, F]
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class StreamingMagDecoder(nn.Module):
    """Streaming version of ``MagDecoder``."""

    def __init__(self, decoder):
        super().__init__()
        self.dense_block = StreamingDenseBlock(decoder.dense_block)
        # up_conv1 is SPConvTranspose2d with time kernel = 1 -> no temporal cache
        self.up_conv1 = decoder.up_conv1
        self.mask_conv = decoder.mask_conv
        self.beta = decoder.beta

    def reset(self):
        self.dense_block.reset()

    def forward(self, x):
        # x: [B, C, 1, F//2]
        x = self.dense_block(x)
        x = self.up_conv1(x)
        x = self.mask_conv(x)
        x = self.beta * torch.sigmoid(x)
        return x


class StreamingPhaseDecoder(nn.Module):
    """Streaming version of ``PhaseDecoder``."""

    def __init__(self, decoder):
        super().__init__()
        self.dense_block = StreamingDenseBlock(decoder.dense_block)
        self.up_conv1 = decoder.up_conv1
        self.phase_conv_r = decoder.phase_conv_r
        self.phase_conv_i = decoder.phase_conv_i

    def reset(self):
        self.dense_block.reset()

    def forward(self, x):
        # x: [B, C, 1, F//2]
        x = self.dense_block(x)
        x = self.up_conv1(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


class StreamingCausalMambaBlock(nn.Module):
    """Streaming version of the unidirectional Mamba block.

    Uses ``Mamba.step`` for a single-frame recurrent update.
    """

    def __init__(self, block):
        super().__init__()
        self.mamba = block.forward_blocks
        self.proj = block.output_proj
        self.norm = block.norm
        self.register_buffer("conv_state", None, persistent=False)
        self.register_buffer("ssm_state", None, persistent=False)

    def reset(self, batch_size):
        self.conv_state, self.ssm_state = self.mamba.allocate_inference_cache(
            batch_size=batch_size,  # batch_size == B * F_enc (temporal Mamba's batch)
            max_seqlen=1,
        )

    def forward(self, x):
        # x: [B, 1, D]
        y, self.conv_state, self.ssm_state = self.mamba.step(
            x, self.conv_state, self.ssm_state
        )
        y = y + x
        y = self.proj(y)
        return self.norm(y)


class StreamingTFMambaBlock(nn.Module):
    """Streaming version of ``TFMambaBlock``.

    Time Mamba is streamed one frame at a time; frequency Mamba is applied in
    full (non-streaming).
    """

    def __init__(self, block):
        super().__init__()
        self.time_mamba = StreamingCausalMambaBlock(block.time_mamba)
        self.freq_mamba = block.freq_mamba  # full-frequency, non-streaming

    def reset(self, batch_size):
        self.time_mamba.reset(batch_size)

    def forward(self, x):
        # x: [B, C, 1, F]
        B, C, _, F = x.shape

        # ---- Time Mamba (true streaming) ----
        x_t = x.permute(0, 3, 2, 1).reshape(B * F, 1, C)  # [B*F, 1, C]
        x_t = self.time_mamba(x_t) + x_t
        x_t = x_t.view(B, F, 1, C).permute(0, 3, 2, 1)  # -> [B, C, 1, F]

        # ---- Frequency Mamba (non-causal, full freq) ----
        x_f = x_t.permute(0, 2, 3, 1).reshape(B, F, C)  # [B, F, C]
        x_f = self.freq_mamba(x_f) + x_f
        x_f = x_f.view(B, 1, F, C).permute(0, 3, 1, 2)  # -> [B, C, 1, F]

        return x_f


class StreamingMambAttention(nn.Module):
    """Streaming wrapper over the non-causal ``MambAttention`` model.

    Reuses the non-causal model's weights. Call ``reset(batch_size)`` before
    streaming a new utterance, then feed one frame at a time. ``batch_size``
    is the batch dimension of the temporal Mamba, i.e. ``B * F_enc`` where
    ``F_enc`` is the frequency-bin count *after* the encoder's freq
    downsampling (``(F + 2) // 2`` for the padded STFT input).
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = StreamingDenseEncoder(model.dense_encoder)
        self.mamba = nn.ModuleList(
            [StreamingTFMambaBlock(b) for b in model.TFMamba]
        )
        self.mag_decoder = StreamingMagDecoder(model.mask_decoder)
        self.pha_decoder = StreamingPhaseDecoder(model.phase_decoder)

    def reset(self, batch_size):
        self.encoder.reset()
        for l in self.mamba:
            l.reset(batch_size)
        self.mag_decoder.reset()
        self.pha_decoder.reset()

    def forward(self, noisy_mag_t, noisy_pha_t):
        """Process a single time frame.

        Args:
            noisy_mag_t: [B, F, 1]
            noisy_pha_t: [B, F, 1]

        Returns:
            mag: [B, F, 1]
            pha: [B, F, 1]
        """
        B, F, _ = noisy_mag_t.shape

        # [B, F, 1] -> [B, 1, 1, F]
        x_mag = noisy_mag_t.permute(0, 2, 1).unsqueeze(1)
        x_pha = noisy_pha_t.permute(0, 2, 1).unsqueeze(1)
        x = torch.cat([x_mag, x_pha], dim=1)  # [B, 2, 1, F]

        # Zero-pad frequency by 2 (mirrors the non-causal generator)
        zeros = torch.zeros(B, x.size(1), 1, 2, device=x.device)
        x = torch.cat([x, zeros], dim=-1)  # [B, 2, 1, F+2]

        # Encoder
        x = self.encoder(x)  # [B, C, 1, (F+2)//2]

        # Mamba blocks
        for b in self.mamba:
            x = b(x)

        # Decoders (truncate the padded frequency bins before masking)
        mag = self.mag_decoder(x)[:, :, :, :F] * x_mag  # [B, 1, 1, F]
        pha = self.pha_decoder(x)[:, :, :, :F]  # [B, 1, 1, F]

        mag = mag.permute(0, 3, 1, 2)  # [B, F, 1, 1]
        pha = pha.permute(0, 3, 1, 2)  # [B, F, 1, 1]

        mag = mag.squeeze(-1)  # [B, F, 1]
        pha = pha.squeeze(-1)  # [B, F, 1]

        return mag, pha
