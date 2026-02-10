## Training Pipeline Overview

**Segmentation Network Training** - Train a 3D UNet on both original and augmented data


### Hardware Requirements
- Python >= 3.7
- CUDA >= 11.0
- GPU: Minimum 4 GPUs (recommended for optimal performance)
- GPU Memory: ~40GB per GPU

### Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```


### Segmentation Network Training

After training the diffusion model and generating augmented data:

```bash
cd segmentation

# Train the 3D segmentation network
python vseg_train.py

# Or with custom configuration
python vseg_train.py -i /path/to/your/config.py
```
Configuration File Located at:
`segmentation/config/config.json`

**Key Features:**
- Loads both original and augmented data
- Trains a 3D UNet for volumetric segmentation
- Uses Weighted Dice loss for optimization
- Supports multi-GPU training

**Training Arguments:**
- `-i, --input`: Configuration file path
- Default configuration uses GPUs 0,1,2,3
- Automatically resumes from checkpoints if available
