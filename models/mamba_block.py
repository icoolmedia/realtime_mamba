"""Pure-Mamba time-frequency blocks for causal speech enhancement.

This is the "方案 A" architecture: the temporal dimension is modeled by a
*unidirectional* Mamba (a fixed-size recurrent state, streamable via
``Mamba.step``), and the frequency dimension is modeled by a *bidirectional*
Mamba (the frequency axis is fixed/non-streaming, so lookahead within a frame
is free). All multi-head attention is removed, eliminating the KV-cache
requirement of the original MambAttention.

Reference: NVIDIA Realtime_REUSE / RT-SEMamba.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class CausalMambaBlock(nn.Module):
    """Unidirectional Mamba block (for the temporal axis).

    x: [B, T, D]
    """

    def __init__(self, d_model, cfg):
        super(CausalMambaBlock, self).__init__()
        d_state = cfg['model_cfg']['d_state']  # 16
        d_conv = cfg['model_cfg']['d_conv']  # 4
        expand = cfg['model_cfg']['expand']  # 4

        self.forward_blocks = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out_fw = self.forward_blocks(x) + x
        out = self.output_proj(out_fw)
        return self.norm(out)


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block (for the frequency axis).

    x: [B, F, D]
    """

    def __init__(self, d_model, cfg):
        super(BiMambaBlock, self).__init__()
        d_state = cfg['model_cfg']['d_state']
        d_conv = cfg['model_cfg']['d_conv']
        expand = cfg['model_cfg']['expand']

        self.forward_blocks = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.backward_blocks = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.output_proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out_fw = self.forward_blocks(x) + x

        out_bw = self.backward_blocks(torch.flip(x, dims=[1])) + torch.flip(x, dims=[1])
        out_bw = torch.flip(out_bw, dims=[1])

        out = torch.cat([out_fw, out_bw], dim=-1)
        out = self.output_proj(out)
        return self.norm(out)


class TFMambaBlock(nn.Module):
    """Temporal-Frequency Mamba block.

    - ``time_mamba``: unidirectional Mamba over the time axis (causal).
    - ``freq_mamba``: bidirectional Mamba over the frequency axis.

    Input/output: [B, C, T, F].
    """

    def __init__(self, cfg):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.time_mamba = CausalMambaBlock(d_model=self.hid_feature, cfg=cfg)
        self.freq_mamba = BiMambaBlock(d_model=self.hid_feature, cfg=cfg)

    def forward(self, x):
        b, c, t, f = x.size()

        # ---- Time Mamba (causal, unidirectional) ----
        # [B, C, T, F] -> [B*F, T, C]
        x_t = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_mamba(x_t) + x_t
        x_t = x_t.view(b, f, t, c).permute(0, 3, 2, 1)  # -> [B, C, T, F]

        # ---- Frequency Mamba (bidirectional) ----
        # [B, C, T, F] -> [B*T, F, C]
        x_f = x_t.permute(0, 2, 3, 1).contiguous().view(b * t, f, c)
        x_f = self.freq_mamba(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)  # -> [B, C, T, F]

        return x_f
