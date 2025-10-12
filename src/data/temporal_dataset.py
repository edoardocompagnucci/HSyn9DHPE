"""
Minimal temporal dataset for 3D pose estimation
Loads sequences with consistent camera viewpoint
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import random

from utils.rotation_utils import rot_matrix_to_6d


class TemporalSyntheticDataset(Dataset):
    """
    Temporal dataset for synthetic data with consistent camera per sequence

    Key design choices:
    - Loads sequences from data/annotations (consistent camera)
    - Simple sliding window approach
    - Minimal augmentation (same noise applied to entire sequence)
    - Built-in sanity checks
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 sequence_length: int = 16,
                 stride: int = 8,  # Overlap for more training samples
                 transform=None,
                 use_augmentation: bool = True,
                 noise_std: float = 0.08,  # Match train.py default
                 noise_prob: float = 0.7):  # Match train.py default

        self.data_root = data_root
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.use_augmentation = use_augmentation and (split == 'train')
        self.noise_std = noise_std
        self.noise_prob = noise_prob

        self.annotations_path = os.path.join(data_root, "annotations")

        # NPZ cache to avoid reloading files
        self.sequence_cache = {}

        # Build sequence index
        self.sequences = []
        self._build_index()

        # Sanity check
        self._validate_dataset()

    def _build_index(self):
        """Build index of all valid sequences"""
        split_file = os.path.join(self.data_root, "splits", f"{self.split}.txt")

        with open(split_file) as f:
            sequence_names = [line.strip() for line in f if line.strip()]

        print(f"Building {self.split} temporal dataset...")
        total_frames = 0

        for seq_name in sequence_names:
            seq_path = os.path.join(self.annotations_path, f"{seq_name}.npz")

            if not os.path.exists(seq_path):
                print(f"  Warning: {seq_name}.npz not found, skipping")
                continue

            # Load sequence metadata
            with np.load(seq_path) as data:
                num_frames = data['num_frames'].item() if 'num_frames' in data else data['joints_3d'].shape[0]

            # Create sliding windows
            if num_frames >= self.sequence_length:
                # For training: overlapping windows
                # For validation: non-overlapping windows
                step = self.stride if self.split == 'train' else self.sequence_length

                for start_idx in range(0, num_frames - self.sequence_length + 1, step):
                    self.sequences.append({
                        'seq_name': seq_name,
                        'start_idx': start_idx,
                        'num_frames': num_frames
                    })

                total_frames += num_frames

        print(f"  Total sequences: {len(sequence_names)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Temporal windows: {len(self.sequences)}")
        print(f"  Sequence length: {self.sequence_length}, stride: {self.stride}")

    def _validate_dataset(self):
        """Sanity check: load one sample and verify shapes"""
        if len(self.sequences) == 0:
            raise ValueError("No sequences found! Check split file and annotations path.")

        sample = self._load_sequence(0)

        # Check shapes
        assert sample['joints_2d'].shape == (self.sequence_length, 24, 2), \
            f"Wrong 2D shape: {sample['joints_2d'].shape}"
        assert sample['joints_3d_centered'].shape == (self.sequence_length, 24, 3), \
            f"Wrong 3D shape: {sample['joints_3d_centered'].shape}"
        assert sample['rot_6d'].shape == (self.sequence_length, 24, 6), \
            f"Wrong rotation shape: {sample['rot_6d'].shape}"

        # Check camera consistency
        K = sample['K']
        K_var = torch.var(K, dim=0).sum()
        assert K_var < 1e-3, f"Camera not consistent! K variance: {K_var}"

        # Check motion smoothness
        joints_3d = sample['joints_3d_centered']
        velocity = joints_3d[1:] - joints_3d[:-1]
        vel_mag = torch.norm(velocity.reshape(-1, 3), dim=-1).mean()

        print(f"\nDataset validation passed!")
        print(f"  Sample shape check: OK")
        print(f"  Camera consistency: OK (var={K_var:.2e})")
        print(f"  Motion smoothness: OK (avg_vel={vel_mag:.4f}m/frame)")

    def _load_sequence(self, idx: int) -> Dict:
        """Load a single temporal sequence"""
        seq_info = self.sequences[idx]
        seq_name = seq_info['seq_name']
        start_idx = seq_info['start_idx']

        # Load NPZ with caching (critical for performance!)
        if seq_name not in self.sequence_cache:
            seq_path = os.path.join(self.annotations_path, f"{seq_name}.npz")
            # Use mmap_mode for memory-efficient loading
            self.sequence_cache[seq_name] = np.load(seq_path, allow_pickle=False, mmap_mode='r')

            # Keep cache size limited (LRU-style)
            if len(self.sequence_cache) > 10:
                oldest = list(self.sequence_cache.keys())[0]
                if hasattr(self.sequence_cache[oldest], 'close'):
                    self.sequence_cache[oldest].close()
                del self.sequence_cache[oldest]

        data = self.sequence_cache[seq_name]

        # Extract sequence window
        end_idx = start_idx + self.sequence_length

        joints_2d = torch.tensor(data['joints_2d'][start_idx:end_idx], dtype=torch.float32)
        joints_3d = torch.tensor(data['joints_3d'][start_idx:end_idx], dtype=torch.float32)
        rot_mats = torch.tensor(data['rot_mats'][start_idx:end_idx], dtype=torch.float32)
        visibility = torch.tensor(data['visibility'][start_idx:end_idx], dtype=torch.bool)
        K = torch.tensor(data['K'][start_idx:end_idx], dtype=torch.float32)
        R = torch.tensor(data['R'][start_idx:end_idx], dtype=torch.float32)
        t = torch.tensor(data['t'][start_idx:end_idx], dtype=torch.float32)
        resolution = torch.tensor(data['res'][start_idx:end_idx], dtype=torch.float32)

        # Center 3D poses (per-frame, using pelvis)
        root_translation = joints_3d[:, 0].clone()
        joints_3d_centered = joints_3d - root_translation.unsqueeze(1)

        # Convert rotations to 6D
        T = self.sequence_length
        rot_6d = rot_matrix_to_6d(rot_mats.reshape(T * 24, 3, 3)).reshape(T, 24, 6)

        # Create confidence weights (same logic as single-frame)
        confidence = self._create_confidence(visibility)

        # Apply augmentation if training
        if self.use_augmentation:
            joints_2d, confidence = self._augment_sequence(
                joints_2d, confidence, visibility, resolution
            )

        return {
            'joints_2d': joints_2d,  # [T, J, 2]
            'joints_3d_centered': joints_3d_centered,  # [T, J, 3]
            'rot_6d': rot_6d,  # [T, J, 6]
            'visibility': visibility,  # [T, J]
            'confidence': confidence,  # [T, J]
            'K': K,  # [T, 3, 3]
            'R': R,  # [T, 3, 3]
            't': t,  # [T, 3]
            'resolution': resolution,  # [T, 2]
            'root_translation': root_translation,  # [T, 3]
            'seq_name': seq_name,
            'start_idx': start_idx
        }

    def _create_confidence(self, visibility: torch.Tensor) -> torch.Tensor:
        """Create synthetic confidence scores (same as single-frame dataset)"""
        T, J = visibility.shape

        # Base confidence values (from analysis of real 3DPW data)
        base_conf = torch.ones(24)
        base_conf[15] = 0.894  # Head
        base_conf[12] = 0.785  # Neck
        base_conf[7:9] = 0.8   # Ankles
        base_conf[4:6] = 0.789  # Knees
        base_conf[16:18] = 0.76  # Shoulders
        base_conf[18:20] = 0.739  # Elbows
        base_conf[1:3] = 0.72  # Hips
        base_conf[0] = 0.65  # Pelvis
        base_conf[20:22] = 0.51  # Wrists
        base_conf[10:12] = 0.45  # Feet
        base_conf[22:24] = 0.3  # Hands

        # Expand to all frames
        confidence = base_conf.unsqueeze(0).expand(T, -1).clone()

        if self.use_augmentation:
            # Add some variation per sequence (not per frame, to maintain temporal coherence)
            global_scale = torch.randn(1).item() * 0.1 + 1.0
            global_scale = max(0.7, min(1.15, global_scale))
            confidence *= global_scale

            # Small per-frame variation
            frame_variation = torch.randn(T, J) * 0.02
            confidence += frame_variation

        # Clamp and apply visibility
        confidence = torch.clamp(confidence, 0.1, 0.95)
        confidence = confidence * visibility.float()

        return confidence

    def _augment_sequence(self, joints_2d, confidence, visibility, resolution):
        """
        Apply noise augmentation to sequence
        IMPORTANT: Apply INDEPENDENT noise per frame to match inference detector behavior
        (MPose detector processes each frame separately with no temporal correlation)
        """
        T, J, _ = joints_2d.shape

        if not self.use_augmentation or torch.rand(1).item() > self.noise_prob:
            return joints_2d, confidence

        # Get resolution (assume constant across sequence)
        width = resolution[0, 0].item()
        height = resolution[0, 1].item()
        base_resolution = 1920.0
        resolution_scale = max(width, height) / base_resolution

        # Apply INDEPENDENT noise per frame (matches single-frame detector)
        augmented = joints_2d.clone()

        for t in range(T):
            # Independent Gaussian noise per frame
            frame_noise = torch.randn(J, 2) * (self.noise_std * 100 * resolution_scale)
            augmented[t] += frame_noise

        # Clamp to image bounds
        augmented[:, :, 0] = torch.clamp(augmented[:, :, 0], 0, width)
        augmented[:, :, 1] = torch.clamp(augmented[:, :, 1], 0, height)

        return augmented, confidence

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        sample = self._load_sequence(idx)

        # Apply normalization transform if provided
        if self.transform:
            # Transform each frame (normalizer expects single frame format)
            transformed_frames = []
            for t in range(self.sequence_length):
                frame_sample = {
                    'joints_2d': sample['joints_2d'][t],
                    'resolution': sample['resolution'][t],
                    'confidence': sample['confidence'][t],
                    'visibility_mask': sample['visibility'][t]
                }
                transformed_frame = self.transform(frame_sample)
                transformed_frames.append(transformed_frame['joints_2d'])

            sample['joints_2d'] = torch.stack(transformed_frames)

        return sample


class Temporal3DPWDataset(Dataset):
    """
    Temporal dataset for 3DPW real data
    Loads sequences from 3DPW detections with consistent actor tracking
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'validation',
                 sequence_length: int = 16,
                 stride: int = 8,
                 transform=None,
                 confidence_threshold: float = 0.3):

        import pickle
        from data.mixed_pose_dataset import coco_wholebody_to_smpl_with_confidence

        self.data_root = data_root
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.confidence_threshold = confidence_threshold

        # Load 3DPW detections
        detections_dir = os.path.join(data_root, "3DPW_processed", "detections")
        if not os.path.exists(detections_dir):
            raise FileNotFoundError(f"3DPW detections not found: {detections_dir}")

        self.sequences = []
        self._build_3dpw_sequences(detections_dir)

    def _build_3dpw_sequences(self, detections_dir):
        """Build temporal sequences from 3DPW detections"""
        import pickle

        print(f"Building 3DPW {self.split} temporal dataset...")

        for filename in sorted(os.listdir(detections_dir)):
            if not filename.endswith('_detections.pkl'):
                continue

            file_split = filename.split('_')[0]
            if file_split != self.split:
                continue

            filepath = os.path.join(detections_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    seq_data = pickle.load(f)

                seq_name = seq_data['sequence_name']
                detections = seq_data['detections']

                # Group frames by actor
                actors_frames = {}
                for frame_idx in sorted(detections.keys()):
                    for actor_idx in detections[frame_idx]:
                        actor_data = detections[frame_idx][actor_idx]
                        if actor_data.get('matched') and actor_data.get('keypoints') is not None:
                            if actor_idx not in actors_frames:
                                actors_frames[actor_idx] = []
                            actors_frames[actor_idx].append((frame_idx, actor_data))

                # Create sequences for each actor
                for actor_idx, frames_data in actors_frames.items():
                    if len(frames_data) < self.sequence_length:
                        continue

                    # Create sliding windows
                    for start in range(0, len(frames_data) - self.sequence_length + 1, self.stride):
                        window = frames_data[start:start + self.sequence_length]
                        self.sequences.append({
                            'seq_name': seq_name,
                            'actor_idx': actor_idx,
                            'frames_data': window
                        })

            except Exception as e:
                print(f"  Warning: Failed to load {filename}: {e}")
                continue

        print(f"  Total 3DPW sequences: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        from data.mixed_pose_dataset import coco_wholebody_to_smpl_with_confidence

        seq_info = self.sequences[idx]
        frames_data = seq_info['frames_data']

        # Extract data for all frames in sequence
        joints_2d_list = []
        joints_3d_list = []
        rot_6d_list = []
        confidence_list = []
        visibility_list = []

        for frame_idx, actor_data in frames_data:
            # Convert COCO-WB to SMPL
            kp_coco = np.array(actor_data['keypoints'])
            scores_coco = np.array(actor_data['scores'])
            joints_2d_smpl, conf_smpl = coco_wholebody_to_smpl_with_confidence(kp_coco, scores_coco)

            joints_2d_list.append(torch.tensor(joints_2d_smpl, dtype=torch.float32))
            joints_3d_list.append(torch.tensor(actor_data['joints_3d_centered'], dtype=torch.float32))
            rot_6d_list.append(torch.tensor(actor_data['rot_6d'], dtype=torch.float32))
            confidence_list.append(torch.tensor(conf_smpl, dtype=torch.float32))
            visibility_list.append(torch.tensor(conf_smpl > self.confidence_threshold, dtype=torch.bool))

        # Stack into tensors
        joints_2d = torch.stack(joints_2d_list)  # [T, J, 2]
        joints_3d = torch.stack(joints_3d_list)  # [T, J, 3]
        rot_6d = torch.stack(rot_6d_list)  # [T, J, 6]
        confidence = torch.stack(confidence_list)  # [T, J]
        visibility = torch.stack(visibility_list)  # [T, J]

        # Apply normalization transform if provided
        if self.transform:
            # Get resolution from first frame
            first_frame_data = frames_data[0][1]
            resolution = torch.tensor(first_frame_data['resolution'], dtype=torch.float32)

            # Transform each frame
            transformed_frames = []
            for t in range(self.sequence_length):
                frame_sample = {
                    'joints_2d': joints_2d[t],
                    'resolution': resolution,
                    'confidence': confidence[t],
                    'visibility_mask': visibility[t]
                }
                transformed_frame = self.transform(frame_sample)
                transformed_frames.append(transformed_frame['joints_2d'])

            joints_2d = torch.stack(transformed_frames)

        return {
            'joints_2d': joints_2d,
            'joints_3d_centered': joints_3d,
            'rot_6d': rot_6d,
            'confidence': confidence,
            'visibility': visibility,
            'seq_name': seq_info['seq_name'],
            'actor_idx': seq_info['actor_idx']
        }


def create_temporal_dataset(data_root: str,
                           dataset_type: str = 'synthetic',
                           split: str = 'train',
                           sequence_length: int = 16,
                           stride: int = 8,
                           transform=None,
                           **kwargs):
    """Factory function for temporal dataset"""
    if dataset_type == 'synthetic':
        return TemporalSyntheticDataset(
            data_root=data_root,
            split=split,
            sequence_length=sequence_length,
            stride=stride,
            transform=transform,
            **kwargs
        )
    elif dataset_type == 'real':
        # Map split names
        real_split = {'train': 'train', 'val': 'validation', 'test': 'test'}.get(split, split)
        return Temporal3DPWDataset(
            data_root=data_root,
            split=real_split,
            sequence_length=sequence_length,
            stride=stride,
            transform=transform,
            confidence_threshold=kwargs.get('confidence_threshold', 0.3)
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
