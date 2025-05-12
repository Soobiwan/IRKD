# IRKD: Knowledge-Distilled Vision Transformer via Inter-Resolution Training

This repository contains the official implementation for **IRKD (Inter-Resolution Knowledge Distillation)**—a novel approach that combines Curriculum Learning (CL), Knowledge Distillation (KD), and Saliency-based Supervision to train compact Vision Transformers more efficiently.

## 🔍 Overview

IRKD progressively trains a student Vision Transformer with increasing input resolution, enabling the model to learn coarse-to-fine representations. Alongside, it distills knowledge from a powerful teacher model using attention-based saliency maps to guide spatial focus.

---

## 📁 Project Structure

## 📁 Project Structure

| File/Folder                | Description |
|---------------------------|-------------|
| `IRKD PAPER`     | IRKD: Knowledge-Distilled Vision Transformer via Inter-Resolution Training Research Paper |
| `Pipeline.png`                   | ViT Tiny Student using Patch Size 4 (Experiment 2.1-2.4) |
| `P2_OUR_MODEL.ipynb`     | Custom Model using Patch Size 2 (Experiment 1.1-1.3) |
| `P2_OUR_MODEL_SAL_MAP.ipynb` | Custom Model using Patch Size 2 (Experiment 1.4) |
| `P2_KD_only`           | ViT Tiny Student using Patch Size 2 (KD) (Experiment 2.2) |
| `P2_KD_CL`           | ViT Tiny Student using Patch Size 2 (KD+CL) (Experiment 2.3) |
| `P2_KD_CL`           | ViT Tiny Student using Patch Size 2 (KD+CL+SL) (Experiment 2.4) |
| `P_4_ALL`                   | ViT Tiny Student using Patch Size 4 (Experiment 2.1-2.4) |


---



## 📖 Paper Overview

**IRKD** (Inter-Resolution Knowledge Distillation) is a unified training framework for compact Vision Transformers that combines:

1. **Curriculum Learning (CL)**  
   We feed the student progressively larger inputs—12×12 → 16×16 → … → 32×32—so it first masters coarse, low-frequency features before refining high-frequency details. This resolution curriculum acts as an implicit regularizer, speeding early convergence and reducing overfitting.

2. **Knowledge Distillation (KD)**  
   A high-capacity ViT-Small teacher (trained from scratch at 32×32 with the same patch size) provides soft logits through a dedicated distillation token. By matching its outputs, the student inherits rich, pre-learned representations, closing most of the accuracy gap despite its much smaller size.

3. **Saliency-Aware Supervision**  
   We extract gradient-based attention maps (saliency) from selected layers of both teacher and student, and add an MSE loss on their spatial “heatmaps.” This guides the student’s focus toward the same informative regions the teacher uses.
ng path. 

For full details—including architecture diagrams, training schedules, and qualitative saliency visualizations—see our full paper.


## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/salman192003/IRKD.git
cd IRKD
