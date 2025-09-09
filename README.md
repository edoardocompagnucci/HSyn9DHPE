# H-6DHPESyn: 6D Human Pose Estimation with Synthetic Data

A GraphFormer-based approach for 2D-to-3D human pose estimation achieving **119mm MPJPE** on the 3DPW dataset using procedurally augmented synthetic data generated from CMU motion capture animations through a custom Houdini pipeline.

## Overview

This project implements a data-efficient approach to 3D human pose estimation by leveraging procedurally augmented synthetic data from CMU motion capture animations. A custom Houdini pipeline enables sophisticated 3D skeleton augmentation and procedural camera generation to create diverse viewpoints and 2D keypoint distributions. The method uses a GraphFormer architecture to lift 2D keypoints to 3D pose and 6D joint rotations.

### Key Features

- **GraphFormer Architecture**: Graph-based transformer for 2D-to-3D pose lifting
- **6D Rotation Prediction**: Direct prediction of 6D rotation representations
- **Procedural Data Augmentation**: Custom Houdini pipeline for 3D skeleton augmentation
- **Procedural Camera Generation**: Automated diverse viewpoint generation
- **Synthetic 2D Keypoint Distribution**: Procedurally generated training data
- **Domain Adaptation**: Effective transfer from synthetic to real data
- **Centralized SMPL System**: Clean, modular skeleton representation

## Performance

- **PA-MPJPE**: 119mm on 3DPW dataset
- **Training Data**: 981,003 synthetic samples from CMU animations
- **Validation**: 3DPW real-world sequences

## Installation

```bash
# Clone repository
git clone https://github.com/edoardocompagnucci/H-6DHPESyn.git
cd H-6DHPESyn

# Install dependencies (environment.yml not included - see requirements below)
# pip install torch torchvision mmpose matplotlib numpy opencv-python tqdm
```

### Requirements
- Python 3.8+
- PyTorch
- MMPose (for 2D pose detection)
- OpenCV
- NumPy, Matplotlib, tqdm

## Usage

**Note**: This repository contains the model architecture and training code. Actual training data is not included due to size constraints.

### Model Architecture
The complete GraphFormer implementation is available for research and development:
```bash
# View model architecture
cd src/models
# See graphformer.py for implementation details
```

### Inference (with trained model)
```bash
cd src
python inference.py path/to/video.mp4 path/to/output/
# Requires: pre-trained model checkpoint
```

### Data Setup (for training)
```bash
# 1. Download CMU mocap data from http://mocap.cs.cmu.edu/
# 2. Process through custom Houdini pipeline (not included)
# 3. Generate synthetic 2D keypoints and 3D poses
# Training uses fully synthetic data generated from Houdini pipeline
```

## Architecture

The GraphFormer model processes 2D keypoints through:
1. **Input Normalization**: 2D keypoint normalization
2. **Graph Convolution**: SMPL skeleton-aware graph processing
3. **Transformer Layers**: Multi-head attention with positional encoding
4. **Dual Output Heads**: 3D positions and 6D rotations

## Dataset Attribution

### CMU Motion Capture Data
This project uses motion capture data from Carnegie Mellon University:
- **Source**: CMU Graphics Lab Motion Capture Database
- **URL**: http://mocap.cs.cmu.edu/
- **Usage**: Procedural augmentation and synthetic 2D keypoint generation through custom Houdini pipeline
- **License**: Free for research use; may be included in commercial products but not resold directly
- **Citation**: The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.

### SMPL Model
The SMPL (Skinned Multi-Person Linear) model is used for pose representation:
- **License**: SMPL Model License (https://smpl.is.tue.mpg.de/modellicense)
- **Citation**: As specified on https://smpl.is.tue.mpg.de
- **Commercial Use**: See LICENSE.md for commercial licensing information

### 3DPW Dataset
Validation performed on the 3D Poses in the Wild dataset:

```bibtex
@inproceedings{vonMarcard2018,
    title = {Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera},
    author = {von Marcard, Timo and Henschel, Roberto and Black, Michael and Rosenhahn, Bodo and Pons-Moll, Gerard},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018},
    month = {sep}
}
```

## Technical Details

### Data Pipeline
1. **CMU Mocap Import**: Load motion capture animations into Houdini
2. **Procedural Augmentation**: Apply 3D skeleton transformations and variations
3. **Procedural Camera Generation**: Create diverse viewpoints through automated camera placement
4. **2D Keypoint Extraction**: Generate synthetic 2D keypoint distributions from multiple viewpoints
5. **Training**: GraphFormer on fully synthetic 2D-3D pairs from Houdini

### SMPL Skeleton (24 joints)
- **Root**: Pelvis (0)
- **Legs**: Hip → Knee → Ankle → Foot chains
- **Spine**: Pelvis → Spine1 → Spine2 → Spine3 → Neck → Head
- **Arms**: Spine3 → Collar → Shoulder → Elbow → Wrist → Hand

### 6D Rotation Representation
Uses continuous 6D rotation representation for stable training:
- **Input**: 2D keypoints (24 joints, 2D coordinates)
- **Output**: 3D positions (24×3) + 6D rotations (24×6) per joint
- **Conversion**: 6D vectors converted to SO(3) rotation matrices
- **Advantage**: No singularities, continuous gradients

## File Structure

```
H-6DHPESyn/
├── src/
│   ├── train.py              # Training script (requires data setup)
│   ├── inference.py          # Video inference (requires trained model)
│   ├── models/
│   │   └── graphformer.py    # GraphFormer architecture ✓
│   ├── data/
│   │   └── mixed_pose_dataset.py  # Dataset handling ✓
│   └── utils/
│       ├── skeleton.py       # SMPL skeleton definition ✓
│       ├── losses.py         # Loss functions ✓
│       ├── rotation_utils.py # 6D rotation utilities ✓
│       └── transforms.py     # Data preprocessing ✓
├── scripts/
│   ├── preprocess_3dpw_detections.py  # 3DPW validation preprocessing
│   ├── make_splits.py        # Data splitting utility
│   └── make_bone_length.py   # Bone length statistics
└── data/meta/               # Essential metadata ✓
```

**✓ = Available in repository**  
**Missing**: Training data, Houdini pipeline, pre-trained models

## Data Attribution

Please note the following attributions when using this work:

- **SMPL Model**: Licensed under SMPL Model License
- **CMU Mocap**: Please cite CMU Graphics Lab when using derived data
- **3DPW Dataset**: Cite the original ECCV'18 paper for validation results

## Citation

If you use this work, please cite:

```bibtex
@software{H6DHPESyn2024,
    title = {H-6DHPESyn: 6D Human Pose Estimation with Synthetic Data},
    author = {Compagnucci, Edoardo},
    year = {2024},
    url = {https://github.com/edoardocompagnucci/H-6DHPESyn}
}
```

## Acknowledgments

The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.

- CMU Graphics Lab for motion capture data
- SMPL team for the human body model
- 3DPW dataset creators for evaluation benchmark
- MMPose library for 2D pose detection