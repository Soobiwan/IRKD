# IRKD: Knowledge Distilled Vision Transformer with Inter-Resolution Training

This repository contains the implementation and experiments for IRKD (Inter-Resolution Knowledge Distillation), a novel approach to knowledge distillation for Vision Transformers that leverages multi-resolution training strategies.

## Project Structure

### Core Implementation Files

- `Experiment_1a.ipynb`: Initial implementation of IRKD with basic knowledge distillation
  - Contains the base ViTWithDistillation model implementation
  - Implements curriculum learning with progressive resolution training
  - Includes teacher-student training pipeline
  - Uses patch size 4 for student model

- `Experiment_1b.ipynb`: Enhanced implementation with saliency-based distillation
  - Extends Experiment_1a with saliency-aware distillation
  - Implements SaliencyDistiller class for feature-level distillation
  - Uses attention-based saliency maps
  - Includes additional loss terms for saliency matching

- `Saliency_patch_size2.ipynb`: Experiments with patch size 2
  - Focuses on analyzing the impact of patch size on model performance
  - Implements patch size 2 for finer-grained feature extraction
  - Includes saliency analysis and visualization
  - Contains ablation studies on patch size effects
 
- `SMALL_P_4_ALL`: Experiments with patch size 4
- `kdonl`: KD only on patch size 2 for student ViT Tiny



### Key Features

- **Inter-Resolution Training**: Progressive training from low to high resolutions
- **Knowledge Distillation**: Teacher-student framework with distillation tokens
- **Saliency-Aware Distillation**: Feature-level distillation using attention maps
- **Curriculum Learning**: Resolution-based curriculum (12px → 32px)
- **Patch Size Analysis**: Experiments with different patch sizes (2 and 4)

### Model Architecture

The implementation includes:
- Teacher Model: DeiT (Data-efficient image Transformers)
- Student Model: Custom ViT with distillation token
- Saliency Distiller: Feature-level distillation using attention maps

### Training Process

1. Teacher Model Training
   - Pre-trained on full resolution
   - Used as knowledge source for student

2. Student Model Training
   - Progressive resolution training (12px → 32px)
   - Knowledge distillation from teacher
   - Saliency-aware feature matching
   - Combined loss: CE + KD + Saliency

## Requirements

- PyTorch
- Transformers (Hugging Face)
- torchvision
- tqdm
- numpy
- matplotlib

## Getting Started

1. Clone this repository
2. Install dependencies
3. Follow the notebooks in sequence:
   - Start with `Experiment_1a.ipynb` for basic implementation
   - Move to `Experiment_1b.ipynb` for saliency-based distillation
   - Use `Saliency_patch_size2.ipynb` for patch size analysis

## Citation

If you use this code in your research, please cite:

(To be added)

## License

(To be added)

## Contact

(To be added)
