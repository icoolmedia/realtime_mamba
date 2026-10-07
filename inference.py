"""Offline (non-causal) inference for the causal MambAttention model."""

import os
import argparse
import yaml
import torch
import librosa
import soundfile as sf

from models.generator import MambAttention
from models.stfts import mag_phase_stft, mag_phase_istft


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

    os.makedirs(args.output_folder, exist_ok=True)

    with torch.no_grad():
        for fname in os.listdir(args.input_folder):
            noisy_wav, _ = librosa.load(
                os.path.join(args.input_folder, fname), sr=sampling_rate
            )
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0))
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)

            noisy_mag, noisy_pha, _ = mag_phase_stft(
                noisy_wav, n_fft, hop_size, win_size, compress_factor, center=False
            )
            mag_g, pha_g, _ = model(noisy_mag, noisy_pha)
            audio_g = mag_phase_istft(
                mag_g, pha_g, n_fft, hop_size, win_size, compress_factor, center=False
            )
            audio_g = audio_g / norm_factor

            out_path = os.path.join(args.output_folder, fname)
            sf.write(out_path, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')
            print(f"Processed {fname}")


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
