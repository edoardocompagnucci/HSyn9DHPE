# HSyn9DHPE: 9D Human Pose Estimation with Houdini Synthetic Data

**HSyn9DHPE** = **H**oudini **Syn**thetic Data for **9D** **H**uman **P**ose **E**stimation
(9D = 3D joint positions + 6D rotation representation)

A graph transformer approach for 2D-to-3D human pose estimation achieving **57.2mm PA-MPJPE** on the 3DPW dataset using Houdini-generated synthetic data from AMASS motion capture sequences.

## Overview

This project implements a data-efficient approach to full pose estimation (3D joint positions + 6D rotation representation) by leveraging synthetic data generated through a custom Houdini pipeline. The pipeline processes AMASS motion capture sequences to create diverse training samples with perfect 3D-2D correspondence through procedural camera generation.

### Key Features

- **Full Pose Estimation**: Predicts both 3D joint positions and 6D rotation representation
- **Synthetic Data Pipeline**: Houdini-based procedural camera generation and data augmentation
- **Root-Centered Coordinate System**: Trained and predicts in pelvis-centered space for consistent pose representation
- **AMASS Motion Data**: High-quality motion capture from 6 diverse datasets
- **End-to-End Inference**: MMPose COCO-WholeBody detection → SMPL mapping → 3D pose lifting
- **Graph Transformer Architecture**: Combines self-attention with graph convolutions for 2D-to-3D pose lifting and 6D rotation prediction
- **3.5M+ Training Samples**: With perfect 3D-2D correspondence

## License

- **Code:** Apache-2.0 (commercial use OK) - see [LICENSE](./LICENSE)
- **Weights:** If/when released, governed by [MODEL_LICENSE.txt](./MODEL_LICENSE.txt)
  - **Non-Commercial Research-Only** (prohibited uses: commercial, surveillance, military, pornographic, defamatory)
  - Includes **takedown/consent clause** for rights/consent issues
  - **No third-party data included** - users must obtain [AMASS](https://amass.is.tue.mpg.de/) and [SMPL](https://smpl.is.tue.mpg.de/) separately
- **Data:** This repo does **not redistribute** AMASS or SMPL data under any circumstances

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
3. **Pose Lifting**: Feed normalized 2D keypoints to graph transformer model
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

# Install MMPose models (required for inference)
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288 --dest checkpoints/
```

### Requirements
- Python 3.10
- PyTorch 2.1.0 with CUDA 11.8
- MMPose ≥1.3.0 (COCO-WholeBody models)
- OpenCV, NumPy, Matplotlib, SciPy

## Usage

### Inference

```bash
cd src
python inference.py path/to/video.mp4 path/to/output/ --checkpoint path/to/checkpoint.pth --smooth_3d
```

The inference script will:
1. Extract frames from video
2. Run MMPose to detect 2D keypoints (COCO-WholeBody format)
3. Map keypoints to SMPL skeleton
4. Predict 3D positions and 6D rotations using the trained model
5. Apply optional 3D smoothing (Savitzky-Golay + SLERP)
6. Save results as JSON files for further processing

## Dataset Attribution

### AMASS Dataset

This project uses motion capture data from the AMASS archive:

```bibtex
@inproceedings{AMASS:2019,
  title={AMASS: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and F. Troje, Nikolaus and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year={2019},
  month = {Oct},
  url = {https://amass.is.tue.mpg.de},
  month_numeric = {10}
}
```

#### ACCAD Dataset
```bibtex
@misc{AMASS_ACCAD,
  title           = {{ACCAD MoCap Dataset}},
  author          = {{Advanced Computing Center for the Arts and Design}},
  url             = {https://accad.osu.edu/research/motion-lab/mocap-system-and-data}
}
```

#### DanceDB Dataset
```bibtex
@article{AMASS_DanceDB,
  author          = {Aristidou, Andreas and Shamir, Ariel and Chrysanthou, Yiorgos},
  title           = {Digital Dance Ethnography: {O}rganizing Large Dance Collections},
  journal         = {J. Comput. Cult. Herit.},
  issue_date      = {January 2020},
  volume          = {12},
  number          = {4},
  month           = nov,
  year            = {2019},
  issn            = {1556-4673},
  articleno       = {29},
  numpages        = {27},
  url             = {https://doi.org/10.1145/3344383},
  doi             = {10.1145/3344383},
  acmid           = {},
  publisher       = {Association for Computing Machinery},
  address         = {New York, NY, USA},
}
```

#### CMU Dataset
```bibtex
@misc{AMASS_CMU,
  title           = {{CMU MoCap Dataset}},
  author          = {{Carnegie Mellon University}},
  url             = {http://mocap.cs.cmu.edu}
}
```

#### KIT Dataset
```bibtex
@inproceedings{AMASS_KIT-CNRS-EKUT-WEIZMANN,
  author          = {Christian Mandery and \"Omer Terlemez and Martin Do and Nikolaus Vahrenkamp and Tamim Asfour},
  title           = {The {KIT} Whole-Body Human Motion Database},
  booktitle       = {International Conference on Advanced Robotics (ICAR)},
  pages           = {329--336},
  year            = {2015},
}

@article{AMASS_KIT-CNRS-EKUT-WEIZMANN-2,
  author          = {Christian Mandery and \"Omer Terlemez and Martin Do and Nikolaus Vahrenkamp and Tamim Asfour},
  title           = {Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion},
  pages           = {796--809},
  volume          = {32},
  number          = {4},
  journal         = {IEEE Transactions on Robotics},
  year            = {2016},
}

@inproceedings{AMASS_KIT-CNRS-EKUT-WEIZMANN-3,
  author          = {Franziska Krebs and Andre Meixner and Isabel Patzer and Tamim Asfour},
  title           = {The {KIT} Bimanual Manipulation Dataset},
  booktitle       = {IEEE/RAS International Conference on Humanoid Robots (Humanoids)},
  pages           = {499--506},
  year            = {2021},
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
      month = oct,
      number = {6},
      pages = {248:1--248:16},
      publisher = {ACM},
      volume = {34},
      year = {2015}
    }
```

### 3DPW Dataset

Evaluation performed on the 3D Poses in the Wild dataset:

```bibtex
@inproceedings{vonMarcard2018,
title = {Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera},
author = {von Marcard, Timo and Henschel, Roberto and Black, Michael and Rosenhahn, Bodo and Pons-Moll, Gerard},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2018},
month = {sep}
}
```

## Architecture

The model uses a **Graph Transformer Encoder** architecture that combines multi-head self-attention with polynomial graph convolutions. Inspired by MeshGraphormer, it features dual prediction heads for 3D positions and 6D rotations, learnable joint embeddings, and operates on joint-level (24 joints) rather than vertex-level representations.

**Key References:**
```bibtex
@inproceedings{lin2021mesh,
  title={Mesh graphormer},
  author={Lin, Kevin and Wang, Lijuan and Liu, Zicheng},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={12939--12948},
  year={2021}
}

@inproceedings{zhao2022graformer,
  title={Graformer: Graph-oriented transformer for 3d pose estimation},
  author={Zhao, Weixi and Wang, Weiqiang and Tian, Yunjie},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={20438--20447},
  year={2022}
}
```

## Citation

If you use this work, please cite:

```bibtex
@software{HSyn9DHPE2025,
  title = {HSyn9DHPE: 9D Human Pose Estimation with Houdini Synthetic Data},
  author = {Compagnucci, Edoardo},
  year = {2025},
  url = {https://github.com/edoardocompagnucci/HSyn9DHPE},
  note = {For non-commercial use only}
}
```

## Acknowledgments

- AMASS team for the motion capture archive
- Individual dataset contributors: ACCAD, CMU, KIT, WEIZMANN, Transitions, DanceDB
- SMPL team for the human body model
- 3DPW dataset creators for evaluation benchmark
- MMPose library for 2D pose detection
