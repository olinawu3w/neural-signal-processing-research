# EEGDfus-Adapt: Subject-Personalised EEG Artefact Removal via LoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/neural-signal-processing-research/blob/main/notebooks/eegdfus_lora_colab_interim_demo.ipynb)

MSc Capstone project (HKUST ARIN 6900, Spring 2026).  
A conditional diffusion model for EEG denoising, personalised to individual noise profiles via **LoRA** (Low-Rank Adaptation) fine-tuning — achieving subject-level improvement with only ~3 % of backbone parameters trainable.

---

## Overview

EEG signals recorded outside a clinical setting are heavily contaminated by physiological artefacts — primarily **ocular (EOG)** and **muscular (EMG)** noise.  
This project adapts the **EEGDfus** backbone (Huang et al., 2025) with lightweight LoRA adapters to personalise denoising for subjects with distinct artefact profiles.

### Architecture

```
Noisy EEG (1 × 512)
       │
       ▼
┌─────────────────────────────────────┐
│   DualBranchDenoisingModel          │
│   ├── CNN branch  (Conv1d × 2)      │──FiLM cross-conditioning
│   └── Transformer (3 × EncoderLayer)│   at each depth
└─────────────────────────────────────┘
       │  wrapped in Conditional DDPM
       │  T=500, linear β schedule
       ▼
Denoised EEG (1 × 512)
```

**LoRA personalisation** injects rank-8 adapters (α=16) into every attention projection (W_Q, W_K, W_V, W_fc) and both FFN linear layers of each Transformer encoder in both branches.  
~120 K trainable parameters per subject vs. 3.99 M total (3.0 %).

---

## Notebook

[`notebooks/eegdfus_lora_colab_interim_demo.ipynb`](notebooks/eegdfus_lora_colab_interim_demo.ipynb)

The notebook runs end-to-end in **Google Colab** (free GPU tier):

| Section | Description |
|---|---|
| Setup & mount | Install deps, mount Google Drive |
| Data preparation | EEGdenoiseNet 80/10/10 split, 11× shuffle augmentation, SNR mixing |
| Backbone architecture | `DualBranchDenoisingModel` + `DDPM` wrapper |
| LoRA injection | `LoRALinear`, `inject_lora()`, parameter freeze |
| Backbone training | 4 000 epochs, batch=512, StepLR |
| LoRA fine-tuning | 200 epochs per subject, CosineAnnealingLR, 4 synthetic profiles |
| Evaluation | RRMSE & CC at 10 SNR levels in [−7, +2] dB |

---

## Dataset

[EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet) — Zhang et al., arXiv:2009.11662 (2021).  
Three `.npy` files required (not included in this repo due to size):

```
data/
├── EEG_all_epochs.npy    # shape (N, 512)
├── EOG_all_epochs.npy
└── EMG_all_epochs.npy
```

Download from the [EEGdenoiseNet GitHub releases](https://github.com/ncclabsustech/EEGdenoiseNet) and place in `data/`, or update `CONFIG['data_dir']` inside the notebook to point at your Google Drive path.

---

## Results

### Backbone — Standard Benchmark (EEGdenoiseNet protocol)

Best checkpoint: `backbone_ep2700.pth`

| Noise type | RRMSE ↓ | CC ↑ |
|---|---|---|
| EOG | 0.336 | 0.949 |
| EMG | 0.488 | 0.837 |

### LoRA Personalisation — Δ RRMSE vs. Generic Backbone

| Subject profile | Noise mix | Generic RRMSE | LoRA RRMSE | Δ RRMSE | Generic CC | LoRA CC |
|---|---|---|---|---|---|---|
| A — Heavy EOG | EOG | 0.3186 | 0.2950 | −0.0236 (−7.4 %) | 0.9580 | 0.9604 |
| B — Light EOG | EOG | 0.1680 | 0.1671 | −0.0009 (−0.6 %) | 0.9881 | 0.9890 |
| C — EMG | EMG | 0.7835 | **0.3490** | **−0.4346 (−55.5 %)** | 0.9207 | 0.9109 |
| D — Mixed | EOG + EMG | 0.3469 | 0.4013 | +0.0544 (+15.7 %) | 0.8325 | 0.8242 |
| **Mean** | | **0.4043** | **0.3031** | **−0.1012 (−25.0 %)** | **0.9248** | **0.9211** |

---

## Environment

### Google Colab (recommended)
Open the notebook badge at the top. All dependencies install automatically in the first cell.

### Local setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## References

- W. Huang et al., "EEGDfus: A Conditional Diffusion Model for Fine-Grained EEG Denoising," *IEEE J. Biomed. Health Inform.*, vol. 29, no. 4, pp. 2557–2569, 2025. DOI: [10.1109/JBHI.2024.3504716](https://doi.org/10.1109/JBHI.2024.3504716)
- E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685, 2021.
- J. Ho, A. Jain, P. Abbeel, "Denoising Diffusion Probabilistic Models," arXiv:2006.11239, 2020.
- H. Zhang et al., "EEGdenoiseNet: A benchmark dataset for deep learning solutions of EEG denoising," arXiv:2009.11662, 2021.
- E. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," arXiv:1709.07871, 2017.
