"""Top-level causal MambAttention generator (pure-Mamba, no attention).

Model pipeline:
    noisy (mag, pha) [B, F, T]
        -> DenseEncoder (causal, freq downsample x2)
        -> num_tfmamba x TFMambaBlock (unidirectional time Mamba + bidirectional freq Mamba)
        -> MagDecoder (mask in [0, beta]) * noisy_mag
        -> PhaseDecoder (atan2)
        -> (denoised_mag, denoised_pha, denoised_com)
"""

import torch
import torch.nn as nn
from einops import rearrange

from .mamba_block import TFMambaBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder


class MambAttention(nn.Module):
    """Causal, streaming-ready MambAttention (pure-Mamba variant)."""

    def __init__(self, cfg):
        super(MambAttention, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg'].get('num_tfmamba', 4)

        self.dense_encoder = DenseEncoder(cfg)
        self.TFMamba = nn.ModuleList(
            [TFMambaBlock(cfg) for _ in range(self.num_tscblocks)]
        )
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        """Forward pass.

        Args:
            noisy_mag: [B, F, T]
            noisy_pha: [B, F, T]

        Returns:
            denoised_mag: [B, F, T]
            denoised_pha: [B, F, T]
            denoised_com: [B, F, T, 2]
        """
        B, F, T = noisy_mag.shape

        # [B, F, T] -> [B, 1, T, F]
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)

        # Concatenate magnitude and phase along channel -> [B, 2, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1)

        # Zero-pad the frequency axis by 2 to make the sub-pixel upsample
        # produce an even number of bins (mirrors Realtime_REUSE). The extra
        # bins are truncated back to F at the output.
        zeros = torch.zeros(B, x.size(1), T, 2, device=x.device)
        x = torch.cat((x, zeros), dim=-1)  # [B, 2, T, F+2]

        # Encode
        x = self.dense_encoder(x)

        # Time-frequency Mamba blocks
        for block in self.TFMamba:
            x = block(x)

        # Decode magnitude (mask) and phase, then truncate the padded bins
        mask = self.mask_decoder(x)[:, :, :, :F]  # [B, 1, T, F]
        denoised_mag = rearrange(
            mask * noisy_mag, 'b c t f -> b f t c'
        ).squeeze(-1)  # [B, F, T]
        denoised_pha = rearrange(
            self.phase_decoder(x)[:, :, :, :F], 'b c t f -> b f t c'
        ).squeeze(-1)  # [B, F, T]

        # Combine into complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha),
             denoised_mag * torch.sin(denoised_pha)),
            dim=-1,
        )
        return denoised_mag, denoised_pha, denoised_com
