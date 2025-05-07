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

## 📖 Paper Overview

**IRKD** (Inter-Resolution Knowledge Distillation) is a unified training framework for compact Vision Transformers that combines:

1. **Curriculum Learning (CL)**  
   We feed the student progressively larger inputs—12×12 → 16×16 → … → 32×32—so it first masters coarse, low-frequency features before refining high-frequency details. This resolution curriculum acts as an implicit regularizer, speeding early convergence and reducing overfitting.

2. **Knowledge Distillation (KD)**  
   A high-capacity ViT-Small teacher (trained from scratch at 32×32 with the same patch size) provides soft logits through a dedicated distillation token. By matching its outputs, the student inherits rich, pre-learned representations, closing most of the accuracy gap despite its much smaller size.

3. **Saliency-Aware Supervision**  
   We extract gradient-based attention maps (saliency) from selected layers of both teacher and student, and add an MSE loss on their spatial “heatmaps.” This guides the student’s focus toward the same informative regions the teacher uses.

### Key Results

| Model / Regime                    | PS=2 Student | PS=4 Student | Teacher (ViT-Small) |
|-----------------------------------|--------------|--------------|----------------------|
| **CE only**                       | 66.9%        | 74.0%        | —                    |
| **+ Curriculum Learning**         | 74.9%        | 79.7%        | —                    |
| **+ KD**                          | 76.7%        | 81.4%        | 82.9%                |
| **+ KD + Curriculum**             | 77.1%        | 83.3%        | 82.9%                |
| **+ KD + CL + Saliency Maps**     | 78.7%        | 84.3%        | 82.9%                |

- A **5 M-param ViT-Tiny (PS=2)** student jumps from **64% → 71.5%** under CL and reaches **81.7%** with full IRKD.  
- A **0.6 M-param custom student (PS=2)** sees similar gains: **66.9% → 74.9%** (CL) → **78.7%** (IRKD).  
- The 4×4-patch variant (PS=4) consistently outperforms PS=2 on CIFAR-10, highlighting the patch-size vs. noise trade-off on small images.

### Why It Matters

- **Efficiency**: Early stages use tiny inputs, drastically reducing compute and memory needs.  
- **Robustness**: Curriculum plus saliency enforces a structured, spatially aware learning path.  
- **Simplicity**: All components (CL, KD, saliency) plug into standard training loops with minimal overhead.

For full details—including architecture diagrams, training schedules, and qualitative saliency visualizations—see our full paper.


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



