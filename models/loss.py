"""Loss functions for the causal MambAttention model.

Includes the anti-wrapping phase losses (IP / GD / IAF) and the
consistency-preserving loss, adapted from the original MambAttention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def anti_wrapping_function(x):
    """Wrap phase differences to [-pi, pi] and take absolute value."""
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_losses(phase_r, phase_g, cfg):
    """Compute IP, GD and IAF phase losses.

    Args:
        phase_r: reference phase [B, F, T]
        phase_g: generated phase [B, F, T]

    Returns:
        (ip_loss, gd_loss, iaf_loss)
    """
    dim_freq = cfg['stft_cfg']['n_fft'] // 2 + 1
    dim_time = phase_r.size(-1)

    # Group-delay matrix (frequency-axis first difference)
    gd_matrix = (
        torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1)
        - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2)
        - torch.eye(dim_freq)
    ).to(phase_g.device)

    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    # Instantaneous-frequency matrix (time-axis first difference)
    iaf_matrix = (
        torch.triu(torch.ones(dim_time, dim_time), diagonal=1)
        - torch.triu(torch.ones(dim_time, dim_time), diagonal=2)
        - torch.eye(dim_time)
    ).to(phase_g.device)

    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))
    return ip_loss, gd_loss, iaf_loss


def consistency_preserving_loss(mag_pred, phase_pred, n_fft, hop_size, win_size,
                                compress_factor=1.0, center=False):
    """Explicit consistency-preserving loss: ||STFT(ISTFT(H')) - H'||^2."""
    device = mag_pred.device
    window = torch.hann_window(win_size, device=device)

    # H' = mag * e^{j phase} (compressed domain)
    H = torch.complex(mag_pred * torch.cos(phase_pred), mag_pred * torch.sin(phase_pred))

    # Decompress to linear magnitude for ISTFT
    mag_lin = torch.pow(mag_pred, 1.0 / compress_factor)
    H_lin = torch.complex(mag_lin * torch.cos(phase_pred), mag_lin * torch.sin(phase_pred))

    wav = torch.istft(H_lin, n_fft=n_fft, hop_length=hop_size,
                      win_length=win_size, window=window, center=center)
    H_proj_lin = torch.stft(wav, n_fft=n_fft, hop_length=hop_size,
                            win_length=win_size, window=window,
                            center=center, pad_mode='reflect',
                            normalized=False, return_complex=True)

    proj_mag = torch.pow(H_proj_lin.abs() + 1e-9, compress_factor)
    proj_pha = torch.angle(H_proj_lin)
    H_proj = torch.complex(proj_mag * torch.cos(proj_pha), proj_mag * torch.sin(proj_pha))

    diff = H_proj - H
    return torch.mean(diff.real ** 2 + diff.imag ** 2)


def wgan_generator_loss(fake_scores):
    """WGAN generator loss: -E[D(fake)].

    Args:
        fake_scores: critic scores on generated samples [B, 1] (or any shape).

    Returns:
        scalar loss.
    """
    return -fake_scores.mean()


def wgan_discriminator_loss(real_scores, fake_scores):
    """WGAN critic loss: E[D(fake)] - E[D(real)].

    Minimizing this (with the critic trained to score real higher) corresponds
    to the Wasserstein-1 objective. For gradient penalty, add it separately.
    """
    return fake_scores.mean() - real_scores.mean()


def feature_matching_loss(real_feats, fake_feats):
    """Feature-matching loss: mean L1 distance of intermediate features.

    Args:
        real_feats: list of tensors from the discriminator on real input.
        fake_feats: list of tensors from the discriminator on fake input.

    Returns:
        scalar loss (mean L1 across all feature maps).
    """
    assert len(real_feats) == len(fake_feats)
    loss = 0.0
    for r, f in zip(real_feats, fake_feats):
        loss = loss + F.l1_loss(f, r.detach())
    return loss / len(real_feats)


def distillation_loss(mag_s, pha_s, com_s, mag_t, pha_t, com_t,
                      w_mag=1.0, w_pha=1.0, w_com=1.0):
    """Output-level distillation loss between student and teacher.

    The teacher and student share the same causal STFT grid (same n_fft / hop /
    center / compress_factor), so their outputs are point-wise aligned and can be
    compared directly with MSE.

    Args:
        mag_s, pha_s, com_s: student magnitude/phase/complex [B, F, T] / [B, F, T, 2].
        mag_t, pha_t, com_t: teacher magnitude/phase/complex.

    Returns:
        scalar loss.
    """
    loss_mag = F.mse_loss(mag_s, mag_t)
    loss_pha = F.mse_loss(pha_s, pha_t)
    loss_com = F.mse_loss(com_s, com_t)
    return w_mag * loss_mag + w_pha * loss_pha + w_com * loss_com
