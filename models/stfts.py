"""Causal Short-Time Fourier Transform utilities.

The key difference from a normal STFT is ``center=False``: no symmetric
padding is applied to the signal, so the analysis/synthesis window is only
anchored at past and current samples. This bounds the algorithmic latency to
one window length (e.g. 400 samples @ 16 kHz = 25 ms) and makes the front-end
causal / streamable.

Magnitude compression is a plain power compression ``|X|^c`` (c = 0.3),
matching the original MambAttention configuration (not the log1p used by
Realtime_REUSE).
"""

import torch
import torch.nn as nn


def mag_phase_stft(y, n_fft, hop_size, win_size, compress_factor=1.0,
                   center=False, addeps=False):
    """Compute magnitude and phase using a causal STFT.

    Args:
        y (torch.Tensor): Input audio signal [..., T] or [B, T].
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float): Magnitude compression exponent.
        center (bool): Must be False for causal/streaming.
        addeps (bool): Add epsilon to magnitude/phase for numerical stability.

    Returns:
        tuple: (mag, pha, com) with shapes [..., F, T], F = n_fft // 2 + 1.
    """
    eps = 1e-10
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(
        y, n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode='reflect',
        normalized=False,
        return_complex=True,
    )

    if not addeps:
        mag = torch.abs(stft_spec)
        pha = torch.angle(stft_spec)
    else:
        real_part = stft_spec.real
        imag_part = stft_spec.imag
        mag = torch.sqrt(real_part.pow(2) + imag_part.pow(2) + eps)
        pha = torch.atan2(imag_part + eps, real_part + eps)

    # Compress the magnitude
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_phase_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0,
                    center=False):
    """Inverse STFT from magnitude and phase (decompress magnitude first).

    Args:
        mag (torch.Tensor): Compressed magnitude [..., F, T].
        pha (torch.Tensor): Phase [..., F, T].
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float): Magnitude compression exponent.
        center (bool): Must be False for causal/streaming.

    Returns:
        torch.Tensor: Reconstructed audio signal.
    """
    mag = torch.pow(mag, 1.0 / compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
    )
    return wav
