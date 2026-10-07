"""Simple dataset for causal MambAttention training.

Loads clean/noisy pairs from JSON lists of wav paths, applies causal STFT.
"""

import os
import json
import random
import torch
import torch.utils.data
import librosa

from models.stfts import mag_phase_stft


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, clean_json, noisy_json, sampling_rate=16000,
                 segment_size=32000, n_fft=400, hop_size=100, win_size=400,
                 compress_factor=0.3, split=True, shuffle=True, device=None):
        self.clean_paths = load_json_file(clean_json)
        self.noisy_paths = load_json_file(noisy_json)
        self.clean_dict = {os.path.basename(p): p for p in self.clean_paths}

        if shuffle:
            random.shuffle(self.noisy_paths)

        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.split = split
        self.device = device

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, index):
        noisy_path = self.noisy_paths[index]
        clean_path = self.clean_dict[os.path.basename(noisy_path)]

        noisy_audio, _ = librosa.load(noisy_path, sr=self.sampling_rate)
        clean_audio, _ = librosa.load(clean_path, sr=self.sampling_rate)

        clean_audio = torch.FloatTensor(clean_audio)
        noisy_audio = torch.FloatTensor(noisy_audio)

        # Normalize by noisy energy
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                start = random.randint(0, clean_audio.size(1) - self.segment_size)
                clean_audio = clean_audio[:, start:start + self.segment_size]
                noisy_audio = noisy_audio[:, start:start + self.segment_size]
            else:
                pad = self.segment_size - clean_audio.size(1)
                clean_audio = torch.nn.functional.pad(clean_audio, (0, pad))
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad))

        clean_mag, clean_pha, clean_com = mag_phase_stft(
            clean_audio, self.n_fft, self.hop_size, self.win_size,
            self.compress_factor, center=False,
        )
        noisy_mag, noisy_pha, _ = mag_phase_stft(
            noisy_audio, self.n_fft, self.hop_size, self.win_size,
            self.compress_factor, center=False,
        )

        return (
            clean_audio.squeeze(),
            clean_mag.squeeze(),
            clean_pha.squeeze(),
            clean_com.squeeze(),
            noisy_mag.squeeze(),
            noisy_pha.squeeze(),
        )
