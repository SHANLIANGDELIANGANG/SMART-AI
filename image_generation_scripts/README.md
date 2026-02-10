
## Training Pipeline Overview

Data Augmentation with Diffusion Model** - Generate synthetic CTA images to augment the training dataset
## System Requirements

### Hardware Requirements
- Python >= 3.7
- CUDA >= 11.0
- GPU: Minimum 4 GPUs (recommended for optimal performance)
- GPU Memory: >40GB per GPU

### Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Training with Diffusion Model

Train the diffusion model to generate synthetic CTA images:

```bash
# Train the conditional diffusion model
sh image_generation_scripts/train_bash.sh
```
Configuration File Located at:
`image_generation_scripts/config/config.json`

**Key Features:**
- Conditional generation based on input conditions
- 3D spatial awareness for volumetric data
- Automatic mixed precision training option
- Comprehensive logging and checkpointing

## Acknowledgements
We gratefully acknowledge the foundational contributions of the following open-source projects, which inspired this repository.
* https://github.com/openai/guided-diffusion