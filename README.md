# HSyn6DHPE: 6D Human Pose Estimation using Houdini-Generated Synthetic Data

A GraphFormer-based approach for 2D-to-3D human pose estimation achieving **60mm PA-MPJPE** on the 3DPW dataset using procedurally augmented synthetic data generated from multi-source motion capture animations through a custom Houdini pipeline.

## Overview

This project implements a data-efficient approach to 3D human pose estimation by leveraging procedurally augmented synthetic data from multiple motion capture sources. A custom Houdini pipeline enables automated skeleton retargeting, anatomical augmentation, and physics-based camera parameter generation to create diverse training samples with perfect 3D-2D correspondence. The method uses a GraphFormer architecture to lift 2D keypoints to 3D pose and 6D joint rotations.

### Key Features

- **End-to-End Pipeline**: MMPose COCO WholeBody detection → 3D pose lifting
- **Multi-Source Motion Data**: CMU MoCap + Mixamo + Rokoko animations
- **Houdini Pipeline**: Automated retargeting and augmentation workflow
- **Physics-Based Cameras**: Realistic viewpoint sampling
- **GraphFormer Architecture**: Graph-based transformer for pose lifting
- **6D Rotation Prediction**: Direct prediction of rotation representations
- **Synthetic Dataset**: 300K+ samples with perfect 3D-2D correspondence

## Performance

- **PA-MPJPE**: 60mm (Procrustes-aligned joint error on 3DPW)
- **MPJPE**: 125mm (joint error on 3DPW)
- **MPJAE**: 35° (mean joint angle error)
- **Validation**: 3DPW real-world sequences
- **Note**: Joint-only evaluation

## Pipeline

### Data Sources
The training dataset uses CMU, Mixamo and Rokoko motion capture data.

### Processing Workflow
1. **Animation Import**: Load motion files from all sources into Houdini
2. **Skeleton Retargeting**: Retarget to standardized SMPL skeleton (24 joints)
3. **Anatomical Augmentation**: Apply bone length variations (±10-25% per joint type)
4. **Camera Sampling**: Generate diverse viewpoints with physics-based constraints
5. **Data Export**: Extract 3D positions, 6D rotations, and projected 2D keypoints

### Camera Strategy
- **Distance**: 1.0-3.5m (optimized for subject visibility)
- **Coverage**: ±42° azimuth, ±20° pan, -35°/+30° tilt
- **Focal Length**: 42.5mm
- **Diversity**: 6.5% artistic low/high angle shots

## Installation

```bash
# Clone repository
git clone https://github.com/edoardocompagnucci/HSyn6DHPE.git
cd HSyn6DHPE

# Create conda environment
conda env create -f environment.yml
conda activate hsyn6dhpe

# Install MMPose models (required for inference)
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288 --dest checkpoints/
```

### Requirements
- Python 3.10
- PyTorch 2.1.0 with CUDA 11.8
- MMPose ≥1.3.0 (with COCO WholeBody models)
- OpenCV, NumPy, Matplotlib, tqdm

## Usage

### Inference
```bash
cd src
python inference.py path/to/video.mp4 path/to/output/
```

## Architecture

The GraphFormer processes 2D keypoints through:
1. **Input encoding** with joint embeddings
2. **Transformer layers** with self-attention and graph convolution
3. **Output heads** for 3D positions and 6D rotations

*Architecture inspired by GraFormer (Zheng et al., 2021) with adaptations for 6D rotation prediction.*

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
- **Commercial Use**: See LICENSE.md for commercial licensing information

```bibtex
@article{SMPL:2015,
    author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
    title = {{SMPL}: A Skinned Multi-Person Linear Model},
    journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
    month = oct,
    number = {6},
    pages = {248:1--248:16},
    publisher = {ACM},
    volume = {34},
    year = {2015}
}
```

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


## Citation

If you use this work, please cite:

```bibtex
@software{HSyn6DHPE2025,
    title = {HSyn6DHPE: 6D Human Pose Estimation using Houdini-Generated Synthetic Data},
    author = {Compagnucci, Edoardo},
    year = {2025},
    url = {https://github.com/edoardocompagnucci/HSyn6DHPE}
}
```

## Acknowledgments

The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.

- CMU Graphics Lab for motion capture data
- SMPL team for the human body model
- 3DPW dataset creators for evaluation benchmark
- MMPose library for 2D pose detection