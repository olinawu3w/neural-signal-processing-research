"""
EEGDfus-Adapt: Subject-Personalised EEG Artefact Removal via LoRA-Adapted Diffusion Models
MSc Capstone — HKUST ARIN 6900 (Spring 2026)

Architecture
* Backbone  : EEGDfus dual-branch denoising network (CNN + Transformer with FiLM cross-conditioning)
              wrapped in a conditional DDPM (T=500 linear schedule).
              Reference: Huang et al., "EEGDfus: A Conditional Diffusion Model for Fine-Grained
              EEG Denoising", IEEE JBHI 29(4):2557–2569, 2025.

* LoRA       : Low-rank adapters (rank=8, α=16) injected into every attention projection
              (W_Q, W_K, W_V, fc) and both FFN linear layers of every Transformer encoder block
              in both branches. ~120 K trainable parameters per subject (3 % of backbone).
              Reference: Hu et al., "LoRA", arXiv:2106.09685, 2021.

* Dataset    : EEGdenoiseNet (Zhang et al., arXiv:2009.11662, 2021).
              80/10/10 train/val/test split; 11× shuffle augmentation; mixed EOG+EMG training.

Usage
# Backbone pre-training
python EEGDfus_Adapt_LoRA.py --mode train_backbone --data_dir ./data --output_dir ./outputs

# LoRA fine-tuning for a subject profile
python EEGDfus_Adapt_LoRA.py --mode train_lora --backbone_ckpt outputs/backbone_ep2700.pth \
                              --data_dir ./data --output_dir ./outputs

# Standard benchmark evaluation
python EEGDfus_Adapt_LoRA.py --mode eval --backbone_ckpt outputs/backbone_ep2700.pth \
                              --data_dir ./data --output_dir ./outputs
"""

from __future__ import annotations

import argparse
import math
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Configuration

DEFAULT_CONFIG: dict = {
    # Paths
    "data_dir":    "./data",
    "output_dir":  "./outputs",
    # Data
    "train_per":       0.8,
    "combin_num":      11,          # shuffle augmentation multiplier
    "train_snr_range": (-7.0, 2.0),
    "seed":            42,
    # Backbone training
    "backbone_epochs": 4000,
    "backbone_batch":  512,
    "backbone_lr":     1e-3,
    "lr_step":         1500,
    "lr_gamma":        0.1,
    "valid_interval":  10,
    # LoRA fine-tuning
    "lora_rank":       8,
    "lora_alpha":      16.0,
    "lora_epochs":     200,
    "lora_batch_size": 64,
    "lora_lr":         5e-4,
    # DDPM
    "diffusion": {
        "beta_start": 1e-4,
        "beta_end":   0.02,
        "num_steps":  500,
        "schedule":   "linear",
    },
}

# Synthetic subject profiles used in personalisation experiments
SUBJECT_PROFILES: dict = {
    "A_heavy_eog":  {"noise_mix": ["EOG"],        "snr_train": (-3.0,  2.0), "snr_test": (-7.0, 2.0)},
    "B_light_eog":  {"noise_mix": ["EOG"],        "snr_train": ( 0.0,  2.0), "snr_test": (-7.0, 2.0)},
    "C_emg":        {"noise_mix": ["EMG"],        "snr_train": (-5.0,  0.0), "snr_test": (-7.0, 2.0)},
    "D_mixed":      {"noise_mix": ["EOG", "EMG"], "snr_train": (-5.0,  2.0), "snr_test": (-7.0, 2.0)},
}

# Model Architecture — EEGDfus dual-branch denoising network

# Transformer hyper-parameters (must match training exactly)
_D_MODEL = 512
_D_FF    = 512
_D_K = _D_V = 64
_N_HEADS = 1


class _Conv1d(nn.Conv1d):
    """Conv1d with Kaiming-normal weight initialisation."""
    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class _PositionalEncoding(nn.Module):
    """Sinusoidal noise-level embedding (scalar → ℝ^dim)."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        enc = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        return torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1).unsqueeze(-1)


class _ScaledDotProductAttention(nn.Module):
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(_D_K)
        return torch.matmul(nn.functional.softmax(scores, dim=-1), V)


class MultiHeadAttention(nn.Module):
    """Single-head attention (n_heads=1) with d_model=512."""
    def __init__(self) -> None:
        super().__init__()
        self.W_Q = nn.Linear(_D_MODEL, _D_K * _N_HEADS, bias=False)
        self.W_K = nn.Linear(_D_MODEL, _D_K * _N_HEADS, bias=False)
        self.W_V = nn.Linear(_D_MODEL, _D_V * _N_HEADS, bias=False)
        self.fc  = nn.Linear(_N_HEADS * _D_V, _D_MODEL,  bias=False)
        self.ln  = nn.LayerNorm(_D_MODEL)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        residual, B = Q, Q.size(0)
        q = self.W_Q(Q).view(B, -1, _N_HEADS, _D_K).transpose(1, 2)
        k = self.W_K(K).view(B, -1, _N_HEADS, _D_K).transpose(1, 2)
        v = self.W_V(V).view(B, -1, _N_HEADS, _D_V).transpose(1, 2)
        ctx = _ScaledDotProductAttention()(q, k, v)
        ctx = ctx.transpose(1, 2).reshape(B, -1, _N_HEADS * _D_V)
        return self.ln(self.fc(ctx) + residual)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → ReLU → Linear with residual + LayerNorm."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(_D_MODEL, _D_FF, bias=False),
            nn.ReLU(),
            nn.Linear(_D_FF, _D_MODEL, bias=False),
        )
        self.ln = nn.LayerNorm(_D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.fc(x) + x)


class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ffn  = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(x, x, x))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for cross-branch conditioning.
    Reference: Perez et al., arXiv:1709.07871, 2017.
    """
    def __init__(self, input_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(condition_dim, input_dim)
        self.beta  = nn.Linear(condition_dim, input_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.gamma(c) * x + self.beta(c)


class DualBranchDenoisingModel(nn.Module):
    """EEGDfus backbone: parallel CNN+Transformer branches with FiLM cross-conditioning.

    Input  : x (B, 1, 512) noisy signal  |  cond (B, 1, 512) reference input
             noise_scale (B, 1) sqrt(ᾱ_t) noise level
    Output : ε̂ (B, 1, 512) predicted noise
    """
    def __init__(self, feats: int = 64) -> None:
        super().__init__()
        conv_block = lambda: nn.Sequential(
            _Conv1d(1, feats, 3, padding=1), _Conv1d(feats, feats, 3, padding=1)
        )
        self.stream_x    = nn.ModuleList([conv_block(), EncoderLayer(), EncoderLayer(), EncoderLayer()])
        self.stream_cond = nn.ModuleList([conv_block(), EncoderLayer(), EncoderLayer(), EncoderLayer()])
        self.embed       = _PositionalEncoding(feats)
        self.bridge      = nn.ModuleList([FiLM(_D_MODEL, 1) for _ in range(4)])
        self.conv_out    = nn.Sequential(_Conv1d(feats, feats, 3, padding=1), _Conv1d(feats, 1, 3, padding=1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, noise_scale: torch.Tensor) -> torch.Tensor:
        noise_embed = self.embed(noise_scale)
        skip_features = []
        for layer, bridge in zip(self.stream_x, self.bridge):
            x = layer(x)
            skip_features.append(bridge(x, noise_embed))
        for skip, layer in zip(skip_features, self.stream_cond):
            cond = layer(cond) + skip
        return self.conv_out(cond)


# LoRA Personalisation

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear.

    Adapted forward:  h = W₀ x + (α/r) B A x
    A ~ Kaiming-uniform, B = 0  →  ΔW = 0 at init (preserves backbone behaviour).
    """
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base    = base
        self.scaling = alpha / rank
        self.lora_A  = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B  = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.lora_A.t() @ self.lora_B.t())


def inject_lora(module: nn.Module, rank: int = 8, alpha: float = 16.0) -> None:
    """Recursively replace Linear layers inside every MHA and FFN with LoRALinear."""
    for child in module.children():
        if isinstance(child, MultiHeadAttention):
            child.W_Q = LoRALinear(child.W_Q, rank, alpha)
            child.W_K = LoRALinear(child.W_K, rank, alpha)
            child.W_V = LoRALinear(child.W_V, rank, alpha)
            child.fc  = LoRALinear(child.fc,  rank, alpha)
        elif isinstance(child, FeedForward):
            child.fc[0] = LoRALinear(child.fc[0], rank, alpha)
            child.fc[2] = LoRALinear(child.fc[2], rank, alpha)
        inject_lora(child, rank, alpha)


def freeze_backbone_except_lora(module: nn.Module) -> None:
    """Freeze everything except LoRA deltas and LayerNorm affine scalars."""
    for name, p in module.named_parameters():
        p.requires_grad = "lora_" in name or name.endswith(".ln.weight") or name.endswith(".ln.bias")


def count_trainable(module: nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# DDPM Wrapper

class DDPM(nn.Module):
    """Conditional DDPM wrapping the EEGDfus backbone.

    Training : predicts noise ε from (x_t, y_noisy, √ᾱ_t); ℓ₁ loss.
    Inference: iterative reverse process over T steps starting from Gaussian noise.
    """
    def __init__(self, backbone: DualBranchDenoisingModel, config: dict, device: torch.device) -> None:
        super().__init__()
        self.model    = backbone
        self.cfg      = config["diffusion"]
        self.device   = device
        self.loss_fn  = nn.L1Loss(reduction="sum")
        self._build_schedule(device)

    def _build_schedule(self, device: torch.device) -> None:
        betas = np.linspace(self.cfg["beta_start"], self.cfg["beta_end"], self.cfg["num_steps"])
        alphas = 1.0 - betas
        ac  = np.cumprod(alphas)
        ac_prev = np.append(1.0, ac[:-1])
        self.sqrt_ac_prev = np.sqrt(np.append(1.0, ac))   # used in p_sample noise level

        _t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        self.register_buffer("betas",          _t(betas))
        self.register_buffer("sqrt_rec_ac",    _t(np.sqrt(1.0 / ac)))
        self.register_buffer("sqrt_rec1m_ac",  _t(np.sqrt(1.0 / ac - 1)))
        self.register_buffer("post_log_var",   _t(np.log(np.maximum(betas * (1 - ac_prev) / (1 - ac), 1e-20))))
        self.register_buffer("post_coef1",     _t(betas * np.sqrt(ac_prev) / (1 - ac)))
        self.register_buffer("post_coef2",     _t((1 - ac_prev) * np.sqrt(alphas) / (1 - ac)))

    # -- forward process (training) ------------------------------------------

    def _q_sample(self, x0: torch.Tensor, sqrt_ac: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return sqrt_ac * x0 + (1 - sqrt_ac ** 2).sqrt() * noise

    def p_losses(self, x0: torch.Tensor, y_noisy: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        T = self.cfg["num_steps"]
        t = np.random.randint(1, T + 1)
        sqrt_ac = torch.FloatTensor(
            np.random.uniform(self.sqrt_ac_prev[t - 1], self.sqrt_ac_prev[t], size=B)
        ).to(x0.device).view(B, 1, 1)
        noise  = torch.randn_like(x0)
        x_t    = self._q_sample(x0, sqrt_ac, noise)
        pred   = self.model(x_t, y_noisy, sqrt_ac.view(B, -1))
        return self.loss_fn(noise, pred)

    def forward(self, x0: torch.Tensor, y_noisy: torch.Tensor) -> torch.Tensor:
        return self.p_losses(x0, y_noisy)

    # -- reverse process (inference) -----------------------------------------

    @torch.no_grad()
    def _p_sample(self, x: torch.Tensor, t: int, cond: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        nl = torch.full((B, 1), float(self.sqrt_ac_prev[t + 1]), device=x.device)
        x_recon = self.sqrt_rec_ac[t] * x - self.sqrt_rec1m_ac[t] * self.model(x, cond, nl)
        mean    = self.post_coef1[t] * x_recon + self.post_coef2[t] * x
        noise   = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return mean + noise * (0.5 * self.post_log_var[t]).exp()

    @torch.no_grad()
    def denoise(self, y_noisy: torch.Tensor) -> torch.Tensor:
        x = torch.randn_like(y_noisy)
        for t in reversed(range(self.cfg["num_steps"])):
            x = self._p_sample(x, t, y_noisy)
        return x


# Dataset & Data Preparation (EEGdenoiseNet standard protocol)

def _get_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a noisy EEG segment and normalise both signals by std(noisy).
    Replicates EEGdenoiseNet data_prepare.py exactly.
    """
    snr  = 10 ** (0.1 * snr_db)
    coef = _get_rms(clean) / (_get_rms(noise) * snr + 1e-8)
    noisy = clean + coef * noise
    std   = float(np.std(noisy)) + 1e-8
    return (noisy / std).astype(np.float32), (clean / std).astype(np.float32)


def _shuffle_signal(arr: np.ndarray, n: int) -> np.ndarray:
    """Concatenate n random permutations of arr (11× augmentation)."""
    return np.vstack([arr[np.random.permutation(len(arr))] for _ in range(n)])


def build_backbone_dataset(
    eeg: np.ndarray,
    noise: np.ndarray,
    combin_num: int = 11,
    train_per: float = 0.8,
    snr_range: tuple[float, float] = (-7.0, 2.0),
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, ...]:
    """Build mixed-noise train/val/test splits following EEGdenoiseNet protocol.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test — each (N, 1, 512)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    eeg   = eeg[rng.permutation(len(eeg))].astype(np.float32)
    noise = noise[rng.permutation(len(noise))].astype(np.float32)
    if len(noise) > len(eeg):
        extra = len(noise) - len(eeg)
        eeg   = np.vstack([eeg[:extra], eeg])
    else:
        eeg = eeg[:len(noise)]

    N        = len(eeg)
    n_train  = round(train_per * N)
    n_val    = (N - n_train) // 2
    n_test   = N - n_train - n_val

    # 11× shuffle augmentation on training split only (anti-overfitting step)
    eeg_tr_aug   = _shuffle_signal(eeg[:n_train], combin_num)
    noise_tr_aug = _shuffle_signal(noise[:n_train], combin_num)
    snr_tr = rng.uniform(*snr_range, size=len(eeg_tr_aug))

    def _make_pairs(eeg_s, noise_s, snr_arr):
        X, y = [], []
        for i in range(len(eeg_s)):
            noisy, clean = _mix_at_snr(eeg_s[i], noise_s[i], snr_arr[i])
            X.append(noisy); y.append(clean)
        return (np.expand_dims(np.array(X, dtype=np.float32), 1),
                np.expand_dims(np.array(y, dtype=np.float32), 1))

    X_tr, y_tr = _make_pairs(eeg_tr_aug, noise_tr_aug, snr_tr)
    snr_va = rng.uniform(*snr_range, size=n_val)
    X_va, y_va = _make_pairs(eeg[n_train:n_train + n_val], noise[n_train:n_train + n_val], snr_va)
    snr_te = np.tile(np.linspace(*snr_range, 10), math.ceil(n_test / 10))[:n_test]
    X_te, y_te = _make_pairs(eeg[n_train + n_val:], noise[n_train + n_val:], snr_te)

    # Shuffle train
    idx = rng.permutation(len(X_tr))
    return X_tr[idx], y_tr[idx], X_va, y_va, X_te, y_te


def build_subject_data(
    subject_cfg: dict,
    eeg: np.ndarray,
    eog: np.ndarray,
    emg: np.ndarray,
    n_train: int = 1200,
    n_test: int = 300,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample EEG + artefact pairs for a synthetic subject profile.

    Subject profiles define the artefact mixture and per-subject SNR range,
    allowing evaluation of LoRA personalisation across different noise regimes.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    def _sample_noise(n: int) -> np.ndarray:
        sources = []
        for src in subject_cfg["noise_mix"]:
            pool = eog if src == "EOG" else emg
            sources.append(pool[rng.choice(len(pool), size=n, replace=True)])
        return np.mean(np.stack(sources), axis=0)

    eeg_tr   = eeg[rng.choice(len(eeg), n_train, replace=True)]
    eeg_te   = eeg[rng.choice(len(eeg), n_test,  replace=True)]
    noise_tr = _sample_noise(n_train)
    noise_te = _sample_noise(n_test)

    snr_tr = rng.uniform(*subject_cfg["snr_train"], size=n_train)
    snr_te = np.tile(np.linspace(*subject_cfg["snr_test"], 10), math.ceil(n_test / 10))[:n_test]

    def _pairs(eeg_s, noise_s, snr_arr):
        X, y = [], []
        for i in range(len(eeg_s)):
            noisy, clean = _mix_at_snr(eeg_s[i], noise_s[i], snr_arr[i])
            X.append(noisy); y.append(clean)
        return (np.expand_dims(np.array(X, np.float32), 1),
                np.expand_dims(np.array(y, np.float32), 1))

    return (*_pairs(eeg_tr, noise_tr, snr_tr), *_pairs(eeg_te, noise_te, snr_te))


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# Evaluation Metrics (EEGdenoiseNet protocol)

def compute_rrmse(pred: np.ndarray, clean: np.ndarray) -> float:
    """Relative RMSE: RMSE(pred, clean) / RMS(clean)."""
    p, c = pred.reshape(-1), clean.reshape(-1)
    return float(np.sqrt(np.mean((p - c) ** 2)) / (np.sqrt(np.mean(c ** 2)) + 1e-12))


def compute_cc(pred: np.ndarray, clean: np.ndarray) -> float:
    """Mean Pearson CC over samples."""
    p = pred.squeeze();  c = clean.squeeze()
    if p.ndim == 1:
        p, c = p[None], c[None]
    return float(np.mean([pearsonr(c[i], p[i])[0] for i in range(len(c))]))


@torch.no_grad()
def run_inference(ddpm: DDPM, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Run full T=500-step DDPM reverse process on X (N, 1, 512)."""
    ddpm.eval()
    out = np.empty_like(X)
    for s in tqdm(range(0, len(X), batch_size), desc="Inference", leave=False):
        e = min(s + batch_size, len(X))
        batch = torch.from_numpy(X[s:e]).to(ddpm.device)
        out[s:e] = ddpm.denoise(batch).cpu().numpy()
    return out


def standard_benchmark_eval(
    ddpm: DDPM,
    eeg: np.ndarray,
    noise: np.ndarray,
    noise_label: str,
    seed: int = 42,
) -> dict:
    """EEGdenoiseNet standard evaluation protocol.
    Evaluates all test samples at each of the 10 fixed SNR levels in [-7, +2] dB.
    """
    rng = np.random.RandomState(seed)
    *_, X_test, y_test = build_backbone_dataset(eeg, noise, combin_num=1, train_per=0.8, rng=rng)
    SNR_GRID = np.linspace(-7.0, 2.0, 10)
    n = len(X_test) // 10 if len(X_test) >= 10 else len(X_test)

    # Re-build test set at each fixed SNR level (all n_test samples per level)
    eeg_r = eeg[rng.permutation(len(eeg))].astype(np.float32)[:len(noise)]
    noise_r = noise[rng.permutation(len(noise))].astype(np.float32)
    N = len(noise_r)
    n_train = round(0.8 * N); n_val = (N - n_train) // 2
    eeg_te = eeg_r[n_train + n_val:]
    noise_te = noise_r[n_train + n_val:]
    n_test = len(eeg_te)

    X_blocks, y_blocks = [], []
    for snr in SNR_GRID:
        Xs, ys = zip(*[_mix_at_snr(eeg_te[j], noise_te[j], snr) for j in range(n_test)])
        X_blocks.append(np.expand_dims(np.array(Xs, np.float32), 1))
        y_blocks.append(np.expand_dims(np.array(ys, np.float32), 1))
    X_all = np.vstack(X_blocks);  y_all = np.vstack(y_blocks)

    pred_all = run_inference(ddpm, X_all)
    rows = []
    for k, snr in enumerate(SNR_GRID):
        s, e = k * n_test, (k + 1) * n_test
        rows.append({
            "SNR_dB": round(snr, 1),
            "RRMSE":  compute_rrmse(pred_all[s:e], y_all[s:e]),
            "CC":     compute_cc(pred_all[s:e], y_all[s:e]),
        })
    mean_rrmse = float(np.mean([r["RRMSE"] for r in rows]))
    mean_cc    = float(np.mean([r["CC"]    for r in rows]))

    print(f"\n{'='*56}")
    print(f"  {noise_label} Benchmark (EEGdenoiseNet standard protocol)")
    print(f"{'='*56}")
    print(f"  {'SNR (dB)':>10}  {'RRMSE':>8}  {'CC':>8}")
    for r in rows:
        print(f"  {r['SNR_dB']:>10.1f}  {r['RRMSE']:>8.4f}  {r['CC']:>8.4f}")
    print(f"  {'Mean':>10}  {mean_rrmse:>8.4f}  {mean_cc:>8.4f}")
    print(f"{'='*56}")
    return {"per_snr": rows, "mean_rrmse": mean_rrmse, "mean_cc": mean_cc}


# Backbone Training

def train_backbone(
    ddpm: DDPM,
    eeg: np.ndarray,
    eog: np.ndarray,
    emg: np.ndarray,
    cfg: dict,
) -> None:
    """Pre-train the EEGDfus backbone on mixed EOG+EMG data for cfg['backbone_epochs'] epochs."""
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = ddpm.device

    rng = np.random.RandomState(cfg["seed"])
    # Build separate EOG and EMG splits, then concatenate for mixed training
    Xe_tr, ye_tr, Xe_va, ye_va, *_ = build_backbone_dataset(eeg, eog, cfg["combin_num"], cfg["train_per"], rng=rng)
    Xm_tr, ym_tr, Xm_va, ym_va, *_ = build_backbone_dataset(eeg, emg, cfg["combin_num"], cfg["train_per"], rng=rng)

    X_tr = np.vstack([Xe_tr, Xm_tr]);  y_tr = np.vstack([ye_tr, ym_tr])
    X_va = np.vstack([Xe_va, Xm_va]);  y_va = np.vstack([ye_va, ym_va])
    idx = rng.permutation(len(X_tr));  X_tr, y_tr = X_tr[idx], y_tr[idx]

    print(f"Train: {X_tr.shape}  Val: {X_va.shape}")

    train_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=cfg["backbone_batch"],
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(EEGDataset(X_va, y_va), batch_size=cfg["backbone_batch"],
                              shuffle=False, drop_last=False)

    opt = Adam(ddpm.parameters(), lr=cfg["backbone_lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"])

    best_val, best_path = float("inf"), os.path.join(cfg["output_dir"], "backbone_best.pth")

    for epoch in range(1, cfg["backbone_epochs"] + 1):
        ddpm.train()
        losses = []
        for X_b, y_b in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = ddpm(X_b, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        tr_loss = float(np.mean(losses))

        if epoch % cfg["valid_interval"] == 0:
            ddpm.eval()
            with torch.no_grad():
                val_loss = float(np.mean([ddpm(X.to(device), y.to(device)).item()
                                          for X, y in val_loader]))
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"epoch": epoch, "model": ddpm.state_dict()}, best_path)
            print(f"Epoch {epoch:04d} | train={tr_loss:.4f} | val={val_loss:.4f} | best={best_val:.4f}")

        if epoch % 100 == 0:
            snap = os.path.join(cfg["output_dir"], f"backbone_ep{epoch:04d}.pth")
            torch.save({"epoch": epoch, "model": ddpm.state_dict()}, snap)

        sched.step()

    torch.save({"epoch": cfg["backbone_epochs"], "model": ddpm.state_dict()},
               os.path.join(cfg["output_dir"], "backbone_final.pth"))
    print("Training complete. Best checkpoint:", best_path)


# ---------------------------------------------------------------------------
# LoRA Fine-Tuning
# ---------------------------------------------------------------------------

def train_lora(
    backbone_ddpm: DDPM,
    eeg: np.ndarray,
    eog: np.ndarray,
    emg: np.ndarray,
    cfg: dict,
) -> None:
    """Fine-tune a LoRA-adapted copy of the backbone for each subject profile."""
    device = backbone_ddpm.device
    results = {}

    for name, subject_cfg in SUBJECT_PROFILES.items():
        print(f"\n{'─'*50}")
        print(f"Subject: {name}  noise={subject_cfg['noise_mix']}")

        # Build subject-specific train/test splits
        X_tr, y_tr, X_te, y_te = build_subject_data(subject_cfg, eeg, eog, emg)

        # Inject LoRA into a fresh copy of the backbone
        model = copy.deepcopy(backbone_ddpm)
        inject_lora(model.model, rank=cfg["lora_rank"], alpha=cfg["lora_alpha"])
        model.to(device)
        freeze_backbone_except_lora(model)
        total, trainable = count_trainable(model)
        print(f"  Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

        # 80/20 train/val split within subject
        n_val = max(1, int(0.2 * len(X_tr)))
        X_tr_fit, y_tr_fit = X_tr[:-n_val], y_tr[:-n_val]
        X_val,    y_val    = X_tr[-n_val:],  y_tr[-n_val:]

        train_loader = DataLoader(EEGDataset(X_tr_fit, y_tr_fit),
                                  batch_size=cfg["lora_batch_size"], shuffle=True)
        val_loader   = DataLoader(EEGDataset(X_val, y_val),
                                  batch_size=cfg["lora_batch_size"], shuffle=False)

        opt   = Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lora_lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["lora_epochs"], eta_min=cfg["lora_lr"] * 0.01)

        best_val_loss = float("inf")
        best_path     = os.path.join(cfg["output_dir"], f"lora_{name}_best.pth")

        for epoch in range(1, cfg["lora_epochs"] + 1):
            model.train()
            ep_losses = []
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                opt.zero_grad()
                loss = model(X_b, y_b); loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); ep_losses.append(loss.item())
            ep_loss = float(np.mean(ep_losses))

            model.eval()
            with torch.no_grad():
                val_loss = float(np.mean([model(X.to(device), y.to(device)).item()
                                          for X, y in val_loader]))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
            sched.step()

            if epoch % 40 == 0 or epoch == 1:
                print(f"  epoch {epoch:03d} | train={ep_loss:.4f} | val={val_loss:.4f}")

        # Evaluate on test set
        model.load_state_dict(torch.load(best_path, map_location=device))

        # Generic backbone metrics (no LoRA)
        generic_pred = run_inference(backbone_ddpm, X_te)
        lora_pred    = run_inference(model,         X_te)

        rr_gen = compute_rrmse(generic_pred, y_te);  cc_gen = compute_cc(generic_pred, y_te)
        rr_lra = compute_rrmse(lora_pred,    y_te);  cc_lra = compute_cc(lora_pred,    y_te)
        delta  = rr_lra - rr_gen

        print(f"  Generic  RRMSE={rr_gen:.4f}  CC={cc_gen:.4f}")
        print(f"  LoRA     RRMSE={rr_lra:.4f}  CC={cc_lra:.4f}  Δ={delta:+.4f} ({100*delta/rr_gen:+.1f}%)")
        results[name] = {"generic_rrmse": rr_gen, "lora_rrmse": rr_lra,
                         "generic_cc": cc_gen, "lora_cc": cc_lra}

    print("\n\nPersonalisation summary:")
    print(f"  {'Subject':<20} {'Generic RRMSE':>14} {'LoRA RRMSE':>12} {'Δ RRMSE':>10}")
    for name, r in results.items():
        d = r["lora_rrmse"] - r["generic_rrmse"]
        print(f"  {name:<20} {r['generic_rrmse']:>14.4f} {r['lora_rrmse']:>12.4f} {d:>+10.4f}")


# CLI Entry Point

def _load_backbone(ckpt_path: str, cfg: dict, device: torch.device) -> DDPM:
    backbone = DualBranchDenoisingModel(feats=64).to(device)
    ddpm     = DDPM(backbone, cfg, device).to(device)
    ckpt     = torch.load(ckpt_path, map_location=device)
    raw      = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    backbone_state = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    if backbone_state:
        backbone.load_state_dict(backbone_state, strict=True)
    else:
        backbone.load_state_dict(raw, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}  (epoch={ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'})")
    return ddpm


def main() -> None:
    parser = argparse.ArgumentParser(description="EEGDfus-Adapt: LoRA-personalised EEG denoising")
    parser.add_argument("--mode", choices=["train_backbone", "train_lora", "eval"],
                        default="eval", help="Execution mode")
    parser.add_argument("--data_dir",      default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output_dir",    default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--backbone_ckpt", default=None,
                        help="Path to backbone checkpoint (required for train_lora / eval)")
    parser.add_argument("--backbone_epochs", type=int, default=DEFAULT_CONFIG["backbone_epochs"])
    parser.add_argument("--lora_epochs",     type=int, default=DEFAULT_CONFIG["lora_epochs"])
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, "data_dir": args.data_dir, "output_dir": args.output_dir,
           "backbone_epochs": args.backbone_epochs, "lora_epochs": args.lora_epochs}
    os.makedirs(cfg["output_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load EEGdenoiseNet data
    eeg = np.load(os.path.join(cfg["data_dir"], "EEG_all_epochs.npy")).astype(np.float32)
    eog = np.load(os.path.join(cfg["data_dir"], "EOG_all_epochs.npy")).astype(np.float32)
    emg = np.load(os.path.join(cfg["data_dir"], "EMG_all_epochs.npy")).astype(np.float32)
    print(f"Data loaded — EEG:{eeg.shape}  EOG:{eog.shape}  EMG:{emg.shape}")

    if args.mode == "train_backbone":
        backbone = DualBranchDenoisingModel(feats=64).to(device)
        ddpm     = DDPM(backbone, cfg, device).to(device)
        n_params = sum(p.numel() for p in backbone.parameters())
        print(f"Backbone parameters: {n_params:,}")
        train_backbone(ddpm, eeg, eog, emg, cfg)

    elif args.mode == "train_lora":
        assert args.backbone_ckpt, "--backbone_ckpt required for train_lora"
        ddpm = _load_backbone(args.backbone_ckpt, cfg, device)
        ddpm.eval()
        train_lora(ddpm, eeg, eog, emg, cfg)

    elif args.mode == "eval":
        assert args.backbone_ckpt, "--backbone_ckpt required for eval"
        ddpm = _load_backbone(args.backbone_ckpt, cfg, device)
        standard_benchmark_eval(ddpm, eeg, eog, "EOG")
        standard_benchmark_eval(ddpm, eeg, emg, "EMG")


if __name__ == "__main__":
    main()
