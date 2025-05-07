# IRKD: Knowledge-Distilled Vision Transformer via Inter-Resolution Training

This repository contains the official implementation for **IRKD (Inter-Resolution Knowledge Distillation)**—a novel approach that combines Curriculum Learning (CL), Knowledge Distillation (KD), and Saliency-based Supervision to train compact Vision Transformers more efficiently.

## 🔍 Overview

IRKD progressively trains a student Vision Transformer with increasing input resolution, enabling the model to learn coarse-to-fine representations. Alongside, it distills knowledge from a powerful teacher model using attention-based saliency maps to guide spatial focus.

---

## 📁 Project Structure

| File/Folder                | Description |
|---------------------------|-------------|
| `Experiment_1a.ipynb`     | Baseline ViT with KD and Curriculum Learning (CL); patch size = 4 |
| `Experiment_1b.ipynb`     | Adds saliency-based supervision on top of KD + CL |
| `Saliency_patch_size2.ipynb` | Full IRKD pipeline using patch size = 2 for finer spatial features |
| `SMALL_P_4_ALL`           | Experiments using ViT-Tiny with patch size = 4 |
| `kdonl`                   | KD-only experiments using patch size = 2 (no CL or saliency) |

---



## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/salman192003/IRKD.git
cd IRKD

### Key Features

- **Inter-Resolution Training**: Progressive training from low to high resolutions
- **Knowledge Distillation**: Teacher-student framework with distillation tokens
- **Saliency-Aware Distillation**: Feature-level distillation using attention maps
- **Curriculum Learning**: Resolution-based curriculum (12px → 32px)
- **Patch Size Analysis**: Experiments with different patch sizes (2 and 4)



