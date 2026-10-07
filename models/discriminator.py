"""Multi-band CNN discriminator (WGAN critic) for the causal MambAttention model.

Aligned with the official Real-time RE-USE GAN stage: the discriminator operates
on the compressed magnitude spectrum and is split into frequency sub-bands. Each
band is processed by a small 2D CNN; the sub-band features are concatenated and
pooled into a single (un-activated) critic score, following the Wasserstein-GAN
formulation.

The discriminator is *not* causal: it sees the full magnitude spectrogram of a
training segment. This is fine because it is only used at training time (the
generator weights remain causal / streaming-ready).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BandConv(nn.Module):
    """Per-band feature extractor: 5 layers of strided 2D convolution."""

    def __init__(self, in_channels=1, dim=32):
        super().__init__()
        self.convs = nn.ModuleList()
        ch = in_channels
        for _ in range(5):
            self.convs.append(
                nn.utils.spectral_norm(
                    nn.Conv2d(ch, dim, kernel_size=(3, 3), stride=(2, 2),
                              padding=(1, 1), bias=False)
                )
            )
            ch = dim
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, return_feats=False):
        feats = []
        for conv in self.convs:
            x = self.leaky(conv(x))
            if return_feats:
                feats.append(x)
        if return_feats:
            return x, feats
        return x


class MultiBandDiscriminator(nn.Module):
    """WGAN critic over frequency sub-bands of the magnitude spectrum.

    Args:
        bands: number of frequency sub-bands to split the spectrogram into.
        dim: base channel count for the per-band CNN.
    """

    def __init__(self, bands=4, dim=32):
        super().__init__()
        self.bands = bands
        self.band_convs = nn.ModuleList(
            [_BandConv(in_channels=1, dim=dim) for _ in range(bands)]
        )

        # Fuse concatenated band features.
        fused_dim = dim * bands
        self.fuse = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(fused_dim, dim, kernel_size=(3, 3), padding=(1, 1),
                          bias=False)
            ),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv2d(dim, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
            ),
        )

    def _split_bands(self, x):
        """Split [B, 1, F, T] into ``bands`` contiguous frequency chunks.

        Each chunk is returned as [B, 1, F_chunk, T].
        """
        B, C, F, T = x.shape
        if F % self.bands != 0:
            # Pad frequency axis so it divides evenly, then drop the padding band.
            pad = self.bands - (F % self.bands)
            x = F.pad(x, (0, 0, 0, pad))
            F = x.shape[2]
        chunk = F // self.bands
        return [x[:, :, i * chunk:(i + 1) * chunk, :] for i in range(self.bands)]

    def forward(self, x, return_feats=False):
        """Forward pass.

        Args:
            x: [B, 1, F, T] compressed magnitude spectrogram.
            return_feats: if True, also return per-band intermediate features
                (a list of lists) for feature-matching loss.

        Returns:
            scores: [B, 1] critic scores.
            feats: (optional) list of per-band feature lists.
        """
        bands = self._split_bands(x)

        band_outs = []
        all_feats = []
        for band_conv, b in zip(self.band_convs, bands):
            if return_feats:
                out, feats = band_conv(b, return_feats=True)
                all_feats.append(feats)
            else:
                out = band_conv(b)
            band_outs.append(out)

        # Concatenate band features along the channel axis (they share H/W after
        # equal strides, but H may differ by 1 due to rounding; average-pool to a
        # common size first to be safe).
        target = band_outs[0].shape[-2:]
        band_outs = [
            F.adaptive_avg_pool2d(b, target) if b.shape[-2:] != target else b
            for b in band_outs
        ]
        fused = torch.cat(band_outs, dim=1)
        out = self.fuse(fused)  # [B, 1, H, W]
        scores = out.mean(dim=(2, 3))  # [B, 1]

        if return_feats:
            # Reshape per-band feats into a flat list of layer features (band-major).
            flat_feats = [f for feats in all_feats for f in feats]
            return scores, flat_feats
        return scores
