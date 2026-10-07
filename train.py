"""Training script for the causal MambAttention model.

Two-stage training aligned with Real-time RE-USE:

- **Stage 1 (regression)**: train with reconstruction losses only, until
  ``stage1_steps``. This maximizes fidelity (posterior-mean estimate).
- **Stage 2 (generative)**: add a WGAN adversarial term (multi-band critic) and
  a distillation term from a frozen, larger causal teacher. The generator is
  refined to restore perceptual details.

The model is trained non-causally (full segments) but with causal padding and
``center=False`` STFT, so its weights are directly reusable by the streaming
wrapper. The discriminator/teacher operate in the spectral domain and do not
affect causality.
"""

import os
import time
import argparse
import copy
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import Dataset
from models.generator import MambAttention
from models.discriminator import MultiBandDiscriminator
from models.stfts import mag_phase_istft
from models.loss import (
    phase_losses, consistency_preserving_loss,
    wgan_generator_loss, wgan_discriminator_loss,
    feature_matching_loss, distillation_loss,
)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_teacher(cfg, device):
    """Load a frozen teacher MambAttention with more TFMamba blocks.

    The teacher shares the same STFT grid as the student, so its outputs are
    point-wise aligned. We build the teacher from the student config but override
    ``num_tfmamba``.
    """
    teacher_cfg = copy.deepcopy(cfg)
    teacher_cfg['model_cfg']['num_tfmamba'] = cfg['training_cfg'].get(
        'teacher_num_tfmamba', 12
    )
    teacher = MambAttention(teacher_cfg).to(device)
    ckpt_path = cfg['training_cfg']['teacher_ckpt']
    ckpt = torch.load(ckpt_path, map_location=device)
    teacher.load_state_dict(ckpt['generator'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def train(args):
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(cfg['env_setting']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['env_setting']['seed'])

    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_size = cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']

    model = MambAttention(cfg).to(device)
    print("Generator parameters:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training_cfg']['learning_rate'],
        betas=(cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2']),
    )

    trainset = Dataset(
        cfg['data_cfg']['train_clean_json'],
        cfg['data_cfg']['train_noisy_json'],
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=n_fft, hop_size=hop_size, win_size=win_size,
        compress_factor=compress_factor, split=True, shuffle=True,
        device=device,
    )
    train_loader = DataLoader(
        trainset, batch_size=cfg['training_cfg']['batch_size'],
        shuffle=True, num_workers=cfg['env_setting']['num_workers'],
        pin_memory=True, drop_last=True,
    )

    loss_w = cfg['training_cfg']['loss']
    os.makedirs(args.exp_path, exist_ok=True)

    # Stage 2 resources are created lazily once stage1_steps is reached.
    stage1_steps = cfg['training_cfg'].get('stage1_steps', float('inf'))
    discriminator = None
    optim_d = None
    teacher = None

    steps = 0
    model.train()
    for epoch in range(cfg['training_cfg']['training_epochs']):
        start = time.time()
        for batch in train_loader:
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch
            clean_audio = clean_audio.to(device)
            clean_mag = clean_mag.to(device)
            clean_pha = clean_pha.to(device)
            clean_com = clean_com.to(device)
            noisy_mag = noisy_mag.to(device)
            noisy_pha = noisy_pha.to(device)

            mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)

            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size,
                                      compress_factor, center=False)

            # Reconstruction losses
            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2
            loss_time = F.l1_loss(clean_audio, audio_g)
            loss_con = consistency_preserving_loss(
                mag_g, pha_g, n_fft, hop_size, win_size, compress_factor,
                center=False,
            ) * 2

            loss_recon = (
                loss_mag * loss_w['magnitude']
                + loss_pha * loss_w['phase']
                + loss_com * loss_w['complex']
                + loss_time * loss_w['time']
                + loss_con * loss_w['consistancy']
            )

            # ---- Stage 2: lazily build discriminator + teacher ----
            if steps >= stage1_steps and discriminator is None:
                discriminator = MultiBandDiscriminator(
                    bands=cfg['disc_cfg']['bands'],
                    dim=cfg['disc_cfg']['dim'],
                ).to(device)
                optim_d = torch.optim.AdamW(
                    discriminator.parameters(),
                    lr=cfg['training_cfg']['learning_rate'],
                    betas=(cfg['training_cfg']['adam_b1'],
                           cfg['training_cfg']['adam_b2']),
                )
                teacher = load_teacher(cfg, device)
                print(f"[Stage 2] started at step {steps}; discriminator + teacher loaded.")

            if discriminator is not None:
                # ---- Discriminator update (WGAN critic) ----
                real_mag = clean_mag.unsqueeze(1)          # [B, 1, F, T]
                fake_mag = mag_g.detach().unsqueeze(1)     # [B, 1, F, T]

                optim_d.zero_grad()
                real_scores = discriminator(real_mag)
                fake_scores = discriminator(fake_mag)
                loss_d = wgan_discriminator_loss(real_scores, fake_scores)
                loss_d.backward()
                optim_d.step()

                # ---- Generator update: recon + adv + fm + distill ----
                optimizer.zero_grad()

                fake_scores_g, fake_feats = discriminator(
                    mag_g.unsqueeze(1), return_feats=True
                )
                with torch.no_grad():
                    _, real_feats = discriminator(real_mag, return_feats=True)

                loss_adv = wgan_generator_loss(fake_scores_g)
                loss_fm = feature_matching_loss(real_feats, fake_feats)

                with torch.no_grad():
                    mag_t, pha_t, com_t = teacher(noisy_mag, noisy_pha)

                loss_kd = distillation_loss(
                    mag_g, pha_g, com_g, mag_t, pha_t, com_t,
                    w_mag=loss_w['distill_mag'],
                    w_pha=loss_w['distill_pha'],
                    w_com=loss_w['distill_com'],
                )

                loss_total = (
                    loss_recon
                    + loss_adv * loss_w['adversarial']
                    + loss_fm * loss_w['feature_matching']
                    + loss_kd * loss_w['distill']
                )

                loss_total.backward()
                optimizer.step()
            else:
                # ---- Stage 1: regression only ----
                optimizer.zero_grad()
                loss_recon.backward()
                optimizer.step()
                loss_total = loss_recon

            if torch.isnan(loss_total).any():
                raise ValueError("NaN in loss")

            if steps % cfg['env_setting']['stdout_interval'] == 0:
                print(
                    f"Steps {steps} | total {loss_total.item():.4f} | "
                    f"mag {loss_mag.item():.4f} | pha {loss_pha.item():.4f} | "
                    f"com {loss_com.item():.4f} | time {loss_time.item():.4f} | "
                    f"con {loss_con.item():.4f}"
                )

            if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                ckpt = {'generator': model.state_dict()}
                if discriminator is not None:
                    ckpt['discriminator'] = discriminator.state_dict()
                torch.save(ckpt, os.path.join(args.exp_path, f'g_{steps:08d}.pth'))

            steps += 1

        print(f"Epoch {epoch + 1} took {int(time.time() - start)}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--exp_path', default='checkpoints')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
