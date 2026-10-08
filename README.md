# 流式因果 MambAttention

全因果、流式（1帧进/1帧出）的单通道语音增强模型，纯 Mamba 架构。

## 架构说明

与原始 MambAttention 相比，本实现做了以下因果化改造（对齐 RT-SEMamba / NVIDIA Realtime_REUSE）：

| 组件 | 原始 MambAttention | 本实现 |
|------|-------------------|-----------------|
| 时间维建模 | 双向 Mamba + 共享 MHSA | **单向 Mamba**（固定循环状态，`Mamba.step` 递推） |
| 频率维建模 | 双向 Mamba + 共享 MHSA | **双向 Mamba**（非流式，无需缓存） |
| 注意力 | 有 MHSA | **全部移除**（消除 KV cache） |
| STFT | `center=True` | **`center=False`**（25ms 算法延迟） |
| 归一化 | `InstanceNorm2d` | **`ChannelLayerNorm`**（C 维，因果安全） |
| 时间轴 padding | 对称 | **因果 padding**（全在左侧） |
| 上采样 | `ConvTranspose2d` | **`SPConvTranspose2d`**（子像素卷积） |

## 文件结构

```
.
├── config.yaml                  # 模型/训练配置
├── models/
│   ├── stfts.py                 # 因果 STFT/iSTFT（center=False）
│   ├── mamba_block.py           # CausalMambaBlock / BiMambaBlock / TFMambaBlock
│   ├── codec_module.py          # ChannelLayerNorm + 因果 padding + SPConvTranspose2d
│   ├── generator.py             # 顶层 MambAttention
│   └── loss.py                  # IP/GD/IAF 相位损失 + 一致性损失
├── streaming/
│   └── streaming_model.py       # 流式 wrapper（状态缓存，1帧进1帧出）
├── dataloader.py
├── train.py                     # 训练（非因果 forward，因果 padding）
├── inference.py                 # 离线推理
└── inference_streaming.py       # 流式推理 + 与离线一致性对比
```

## 环境要求

需安装 `mamba_ssm`（推荐 `1.2.x`，需包含 `Mamba.step()` 和
`allocate_inference_cache()` 接口）以及 `torch`、`librosa`、`soundfile`、
`einops`、`pyyaml`。

> 注意：本目录所在的基础 Python 环境未安装 `mamba_ssm`。请在
> MambAttention 的 conda 环境（`mamba_ssm 1.2.2`）中运行训练/推理。

## 训练

```bash
python train.py --config config.yaml --exp_path checkpoints
```

模型以非因果方式（整段 forward）训练，但由于使用了因果 padding 和
`center=False`，其权重可直接被流式 wrapper 复用。

## 推理

离线（整段）推理：

```bash
python inference.py \
    --input_folder noisy/ \
    --output_folder results/ \
    --checkpoint_file checkpoints/g_00000500.pth
```

流式（1帧进1帧出）推理，并验证与离线结果逐帧一致：

```bash
python inference_streaming.py \
    --input_folder noisy/ \
    --output_folder results/ \
    --checkpoint_file checkpoints/g_00000500.pth
```

`inference_streaming.py` 会输出：

- `max |mag diff|` / `max |pha diff|`：流式 vs 离线（因果）的逐帧差异，应接近 0。
- `avg latency/frame`：每帧平均处理耗时。
- `RTF`：实时因子（处理时间 / 音频时长）。

## 流式状态管理

流式 wrapper 通过三类状态缓存实现 1帧进1帧出（`register_buffer(..., persistent=False)`）：

1. `StreamingCausalConv2d.cache` — 时间轴卷积的过去帧缓存。
2. `StreamingCausalMambaBlock.conv_state` — SSM 前深度 1D 卷积的缓存。
3. `StreamingCausalMambaBlock.ssm_state` — 选择性状态空间模型的循环隐状态。

每段新音频前调用 `streaming_model.reset(freq_dim)` 重置所有状态。
