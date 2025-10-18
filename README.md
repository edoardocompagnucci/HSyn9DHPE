# HSyn9DHPE: 3D Position + 6D Rotation Human Pose Estimation

**HSyn9DHPE** = Houdini Synthetic 3D position (3D) + 6D rotation representation (6D) = 9D

A GraphFormer-based approach for 2D-to-3D human pose estimation achieving **57.2mm PA-MPJPE** on the 3DPW dataset using Houdini-generated synthetic data from AMASS motion capture sequences.

## Overview

This project implements a data-efficient approach to full pose estimation (3D joint positions + 6D rotation representation) by leveraging synthetic data generated through a custom Houdini pipeline. The pipeline processes AMASS motion capture sequences to create diverse training samples with perfect 3D-2D correspondence through procedural camera generation.

### Key Features

- **Full Pose Estimation**: Predicts both 3D joint positions and 6D rotation representation
- **Synthetic Data Pipeline**: Houdini-based procedural camera generation and data augmentation
- **Root-Centered Training**: Model trained in root-centered coordinate space
- **AMASS Motion Data**: High-quality motion capture from 6 diverse datasets
- **End-to-End Inference**: MMPose COCO-WholeBody detection → SMPL mapping → 3D pose lifting
- **GraphFormer Architecture**: Graph-based transformer for pose lifting
- **300K+ Training Samples**: With perfect 3D-2D correspondence

## Performance

Evaluated on 3DPW dataset:
- **PA-MPJPE**: 57.2mm (Procrustes-aligned mean per-joint position error)
- **MPJPE**: 120mm (mean per-joint position error)
- **MPJAE**: 36° (mean per-joint angle error)

## Pipeline

### Data Generation

1. **AMASS Sequences**: Load motion capture data from 6 AMASS datasets (ACCAD, CMU, Transitions, DanceDB, KIT, WEIZMANN)
2. **Houdini Processing**:
   - Read joint rotation angles from AMASS sequences
   - Apply rotations to base SMPL skeleton (24 joints)
   - Generate procedural camera views with realistic parameters
3. **Projection & Export**: Project 3D joints to 2D using camera parameters and export .npz files containing:
   - 3D joint positions (SMPL skeleton)
   - 2D joint positions (pixel coordinates)
   - Joint rotations (3x3 rotation matrices, Houdini default)
   - Camera intrinsics and extrinsics
   - Joint visibility flags
   - Image resolution

### Camera Generation Strategy

The Houdini pipeline generates diverse synthetic cameras using procedural sampling:
- **Distance**: Primarily 1.2-3.8m with occasional extreme views (0.5-6.5m)
- **Viewing Angles**: Wide coverage with azimuth, pan, and tilt variation
- **Focal Length**: Distance-correlated (16-135mm) for realistic framing
- **Artistic Shots**: 5% extreme low/high angle perspectives

### Inference Pipeline

1. **2D Detection**: Run MMPose RTMPose on input video to extract COCO-WholeBody keypoints
2. **Keypoint Mapping**: Map COCO-WholeBody joints to SMPL skeleton (24 joints)
3. **Pose Lifting**: Feed normalized 2D keypoints to GraphFormer model
4. **Prediction**: Output 3D joint positions (root-centered) and 6D rotation representation
5. **Post-Processing**: Convert 6D representation to 3x3 rotation matrices
6. **Optional Smoothing**: Apply Savitzky-Golay smoothing for positions and SLERP for rotations

**Outputs:**
- `results_frame.json`: Complete pose data with 3D positions, 6D rotations, and 3x3 rotation matrices
- `houdini_frames/`: Per-frame JSON files for Houdini import
- `visualizations_3d/`: Optional 3D pose visualizations (with `--visualize_3d` flag)

## Installation

```bash
# Clone repository
git clone https://github.com/edoardocompagnucci/HSyn9DHPE.git
cd HSyn9DHPE

# Create conda environment
conda env create -f environment.yml
conda activate hsyn9dhpe

# Install mmcv (required for mmpose)
mim install mmcv==2.1.0

# Download MMPose models (required for inference)
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288 --dest checkpoints/
```

### Requirements
- Python 3.8
- PyTorch 2.0.1 with CUDA 11.8
- MMPose ≥1.3.0 (COCO-WholeBody models)
- mmcv 2.1.0, mmengine 0.8.4
- OpenCV, NumPy, Matplotlib, SciPy

**Note:** The environment includes Visual C++ runtime packages (vc, vs2015_runtime) required for PyTorch on Windows.

## Usage

### Inference

```bash
python src/inference.py path/to/video.mp4 path/to/output/ --checkpoint checkpoints/best_model_57mm.pth --smooth_3d
```

**Additional options:**
- `--visualize_3d`: Generate 3D pose visualization images
- `--smooth_detections`: Smooth 2D detections to remove outliers
- `--skip_frames N`: Process every Nth frame (default: 1)
- `--max_frames N`: Limit processing to N frames

The inference script will:
1. Extract frames from video
2. Run MMPose to detect 2D keypoints (COCO-WholeBody format)
3. Map keypoints to SMPL skeleton
4. Predict 3D positions and 6D rotations using the trained model
5. Apply 3D smoothing (Savitzky-Golay for positions + SLERP for rotations)
6. Save results as JSON files for further processing

## Dataset Attribution

### AMASS Dataset

This project uses motion capture data from the AMASS archive:

```bibtex
@inproceedings{AMASS:ICCV:2019,
  title = {{AMASS}: Archive of Motion Capture as Surface Shapes},
  author = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {International Conference on Computer Vision},
  pages = {5442--5451},
  year = {2019}
}
```

#### ACCAD Dataset
```bibtex
@misc{AMASS_ACCAD,
  title = {{ACCAD MoCap Dataset}},
  author = {{Advanced Computing Center for the Arts and Design}},
  url = {https://accad.osu.edu/research/motion-lab/mocap-system-and-data}
}
```
License: https://creativecommons.org/licenses/by/3.0/

#### CMU Dataset
```bibtex
@misc{AMASS_CMU,
  title = {{CMU MoCap Dataset}},
  author = {{Carnegie Mellon University}},
  url = {http://mocap.cs.cmu.edu}
}
```
License: Free for research use; may be included in commercial products but not resold directly. The database was created with funding from NSF EIA-0196217.

#### KIT Dataset
```bibtex
@inproceedings{AMASS_KIT_1,
  author = {Christian Mandery and \"Omer Terlemez and Martin Do and Nikolaus Vahrenkamp and Tamim Asfour},
  title = {The {KIT} Whole-Body Human Motion Database},
  booktitle = {International Conference on Advanced Robotics (ICAR)},
  pages = {329--336},
  year = {2015}
}

@article{AMASS_KIT_2,
  author = {Christian Mandery and \"Omer Terlemez and Martin Do and Nikolaus Vahrenkamp and Tamim Asfour},
  title = {Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion},
  journal = {IEEE Transactions on Robotics},
  volume = {32},
  number = {4},
  pages = {796--809},
  year = {2016}
}

@inproceedings{AMASS_KIT_3,
  author = {Franziska Krebs and Andre Meixner and Isabel Patzer and Tamim Asfour},
  title = {The {KIT} Bimanual Manipulation Dataset},
  booktitle = {IEEE/RAS International Conference on Humanoid Robots (Humanoids)},
  pages = {499--506},
  year = {2021}
}
```

#### WEIZMANN Dataset
Uses the same citations as KIT dataset (joint KIT-CNRS-EKUT-WEIZMANN collaboration).

### SMPL Model

```bibtex
@article{SMPL:2015,
  author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
  title = {{SMPL}: A Skinned Multi-Person Linear Model},
  journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
  volume = {34},
  number = {6},
  pages = {248:1--248:16},
  year = {2015}
}
```
License: https://smpl.is.tue.mpg.de/modellicense

### 3DPW Dataset

Evaluation performed on the 3D Poses in the Wild dataset:

```bibtex
@inproceedings{vonMarcard2018,
  title = {Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera},
  author = {von Marcard, Timo and Henschel, Roberto and Black, Michael and Rosenhahn, Bodo and Pons-Moll, Gerard},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2018}
}
```

## Architecture

The model uses a GraphFormer architecture adapted for 6D rotation prediction:

```bibtex
@inproceedings{zhao2022graformer,
  title={GraFormer: Graph Convolution Transformer for 3D Pose Estimation},
  author={Zhao, Weixi and Tian, Yunjie and Ye, Qixiang and Jiao, Jianbin and Wang, Weiqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={20438--20447},
  year={2022}
}
```

## Citation

If you use this work, please cite:

```bibtex
@software{HSyn9DHPE2025,
  title = {HSyn9DHPE: 3D Position + 6D Rotation Human Pose Estimation},
  author = {Compagnucci, Edoardo},
  year = {2025},
  url = {https://github.com/edoardocompagnucci/HSyn9DHPE}
}
```

## Acknowledgments

- AMASS team for the motion capture archive
- Individual dataset contributors: ACCAD, CMU, KIT, WEIZMANN, Transitions, DanceDB
- SMPL team for the human body model
- 3DPW dataset creators for evaluation benchmark
- MMPose library for 2D pose detection
