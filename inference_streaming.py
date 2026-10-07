"""Streaming (1-frame-in / 1-frame-out) inference.

Runs the causal MambAttention model in an online fashion, propagating the
recurrent Mamba state and convolution caches. Also performs an offline
inference for comparison and verifies that streaming output matches the
offline (causal) output frame by frame.
"""

import os
import argparse
import time
import yaml
import torch
import librosa
import soundfile as sf

from models.generator import MambAttention
from models.stfts import mag_phase_stft, mag_phase_istft
from streaming.streaming_model import StreamingMambAttention


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def inference(args):
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_size = cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = MambAttention(cfg).to(device)
    ckpt = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(ckpt['generator'])
    model.eval()

    streaming_model = StreamingMambAttention(model).to(device).eval()

    os.makedirs(args.output_folder, exist_ok=True)

    for fname in os.listdir(args.input_folder):
        noisy_wav, _ = librosa.load(
            os.path.join(args.input_folder, fname), sr=sampling_rate
        )
        noisy_wav = torch.FloatTensor(noisy_wav).to(device)

        norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0))
        noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)  # [1, T]

        noisy_mag, noisy_pha, _ = mag_phase_stft(
            noisy_wav, n_fft, hop_size, win_size, compress_factor, center=False
        )  # [1, F, T]

        F, T = noisy_mag.shape[1], noisy_mag.shape[2]

        # ---- Offline (causal) inference ----
        with torch.no_grad():
            mag_off, pha_off, _ = model(noisy_mag, noisy_pha)

        # ---- Streaming inference ----
        # 时间维 Mamba 的循环状态 batch 维度 = B * F_enc，而不是原始 STFT 的 F。
        # encoder 对频率先 pad(+2) 再 stride-2 下采样，故 F_enc = (F + 2) // 2。
        B = noisy_mag.shape[0]
        streaming_model.reset(B * ((F + 2) // 2))
        mag_stream = []
        pha_stream = []
        times = []

        with torch.no_grad():
            for t in range(T):
                start = time.perf_counter()
                mag_t, pha_t = streaming_model(
                    noisy_mag[:, :, t:t + 1], noisy_pha[:, :, t:t + 1]
                )  # [1, F, 1]
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                mag_stream.append(mag_t)
                pha_stream.append(pha_t)

        mag_stream = torch.cat(mag_stream, dim=-1)  # [1, F, T]
        pha_stream = torch.cat(pha_stream, dim=-1)

        # ---- Consistency check ----
        mag_diff = (mag_off - mag_stream).abs().max().item()
        pha_diff = (pha_off - pha_stream).abs().max().item()
        print(f"[{fname}] max |mag diff| = {mag_diff:.6f}, max |pha diff| = {pha_diff:.6f}")

        # ---- Reconstruct audio ----
        audio_g = mag_phase_istft(
            mag_stream, pha_stream, n_fft, hop_size, win_size,
            compress_factor, center=False,
        )
        audio_g = audio_g / norm_factor

        out_path = os.path.join(args.output_folder, fname)
        sf.write(out_path, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')

        total_audio_time = noisy_wav.shape[-1] / sampling_rate
        avg_latency_ms = sum(times) / len(times) * 1000
        rtf = sum(times) / total_audio_time
        print(f"[{fname}] avg latency/frame = {avg_latency_ms:.3f} ms, RTF = {rtf:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint_file', required=True)
    args = parser.parse_args()
    inference(args)


if __name__ == '__main__':
    main()
