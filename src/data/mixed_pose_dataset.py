import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Union
import random

from utils.rotation_utils import rot_matrix_to_6d


def coco_wholebody_to_smpl_with_confidence(kp2d_coco_wb, conf_coco_wb):
    """Convert COCO-WholeBody 133 keypoints to SMPL 24 joints WITH confidence scores"""
    out = np.zeros((24, 2), dtype=kp2d_coco_wb.dtype)
    out_conf = np.zeros(24, dtype=np.float32)
    
    def safe_confidence(*indices, mode='harmonic'):
        """Conservative confidence for interpolated joints"""
        valid_confs = [conf_coco_wb[idx] for idx in indices if conf_coco_wb[idx] > 0]
        
        if not valid_confs:
            return 0.0
        
        if mode == 'harmonic':
            return len(valid_confs) / sum(1/(c + 1e-8) for c in valid_confs)
        elif mode == 'geometric':
            return np.prod(valid_confs) ** (1/len(valid_confs))
        elif mode == 'minimum':
            return min(valid_confs)
        else:
            return np.mean(valid_confs)
    
    hip_center = (kp2d_coco_wb[11] + kp2d_coco_wb[12]) / 2
    shoulder_center = (kp2d_coco_wb[5] + kp2d_coco_wb[6]) / 2
    nose = kp2d_coco_wb[0]
    
    # PELVIS
    out[0] = hip_center
    out_conf[0] = safe_confidence(11, 12, mode='minimum')
    
    # HIPS
    out[1] = kp2d_coco_wb[11]
    out_conf[1] = conf_coco_wb[11]
    out[2] = kp2d_coco_wb[12]
    out_conf[2] = conf_coco_wb[12]
    
    # KNEES & ANKLES
    out[4] = kp2d_coco_wb[13]
    out_conf[4] = conf_coco_wb[13]
    out[5] = kp2d_coco_wb[14]
    out_conf[5] = conf_coco_wb[14]
    out[7] = kp2d_coco_wb[15]
    out_conf[7] = conf_coco_wb[15]
    out[8] = kp2d_coco_wb[16]
    out_conf[8] = conf_coco_wb[16]
    
    # SPINE CHAIN
    spine_base = out[0]
    spine_top = shoulder_center
    spine_vec = spine_top - spine_base
    
    base_spine_conf = safe_confidence(5, 6, 11, 12, mode='harmonic')
    
    out[3] = spine_base + spine_vec * 0.30
    out_conf[3] = base_spine_conf * 0.9
    
    out[6] = spine_base + spine_vec * 0.60
    out_conf[6] = base_spine_conf * 0.8
    
    out[9] = spine_base + spine_vec * 0.75
    out_conf[9] = base_spine_conf * 0.7
    
    # NECK & HEAD
    out[12] = (shoulder_center + nose) * 0.5
    out_conf[12] = safe_confidence(0, 5, 6, mode='harmonic')
    
    out[15] = nose
    out_conf[15] = conf_coco_wb[0]
    
    # SHOULDERS
    out[16] = kp2d_coco_wb[5]
    out_conf[16] = conf_coco_wb[5]
    out[17] = kp2d_coco_wb[6]
    out_conf[17] = conf_coco_wb[6]
    
    # COLLARS
    out[13] = (out[9] + out[16]) * 0.5
    out_conf[13] = min(out_conf[9], out_conf[16]) * 0.8
    
    out[14] = (out[9] + out[17]) * 0.5
    out_conf[14] = min(out_conf[9], out_conf[17]) * 0.8
    
    # ARMS
    out[18] = kp2d_coco_wb[7]
    out_conf[18] = conf_coco_wb[7]
    out[19] = kp2d_coco_wb[8]
    out_conf[19] = conf_coco_wb[8]
    out[20] = kp2d_coco_wb[9]
    out_conf[20] = conf_coco_wb[9]
    out[21] = kp2d_coco_wb[10]
    out_conf[21] = conf_coco_wb[10]
    
    # FEET
    if conf_coco_wb[17] > 0:
        out[10] = kp2d_coco_wb[17]
        out_conf[10] = conf_coco_wb[17]
    else:
        out[10] = out[7] + np.array([0, 20])
        out_conf[10] = min(out_conf[7] * 0.3, 0.2)
    
    if conf_coco_wb[20] > 0:
        out[11] = kp2d_coco_wb[20]
        out_conf[11] = conf_coco_wb[20]
    else:
        out[11] = out[8] + np.array([0, 20])
        out_conf[11] = min(out_conf[8] * 0.3, 0.2)
    
    # HANDS
    left_hand_indices = list(range(91, 112))
    left_hand_confs = conf_coco_wb[left_hand_indices]
    valid_left = left_hand_confs > 0.3
    
    if valid_left.any():
        valid_kpts = kp2d_coco_wb[left_hand_indices][valid_left]
        out[22] = valid_kpts.mean(axis=0)
        out_conf[22] = left_hand_confs[valid_left].mean() * 0.8
    else:
        forearm_vec = out[20] - out[18]
        hand_length = np.linalg.norm(forearm_vec) * 0.2
        if hand_length > 0:
            out[22] = out[20] + (forearm_vec / np.linalg.norm(forearm_vec)) * hand_length
        else:
            out[22] = out[20] + np.array([0, 15])
        out_conf[22] = min(out_conf[20] * 0.2, 0.15)
    
    right_hand_indices = list(range(112, 133))
    right_hand_confs = conf_coco_wb[right_hand_indices]
    valid_right = right_hand_confs > 0.3
    
    if valid_right.any():
        valid_kpts = kp2d_coco_wb[right_hand_indices][valid_right]
        out[23] = valid_kpts.mean(axis=0)
        out_conf[23] = right_hand_confs[valid_right].mean() * 0.8
    else:
        forearm_vec = out[21] - out[19]
        hand_length = np.linalg.norm(forearm_vec) * 0.2
        if hand_length > 0:
            out[23] = out[21] + (forearm_vec / np.linalg.norm(forearm_vec)) * hand_length
        else:
            out[23] = out[21] + np.array([0, 15])
        out_conf[23] = min(out_conf[21] * 0.2, 0.15)
    
    confidence_scaling = {
        22: 0.5, 23: 0.5, 20: 0.7, 21: 0.7, 10: 0.7, 11: 0.7
    }
    
    for joint_idx, scale in confidence_scaling.items():
        out_conf[joint_idx] *= scale
    
    return out, out_conf


class SyntheticPoseAdapter(Dataset):
    """Adapter for synthetic pose data with comprehensive augmentation"""
    
    def __init__(self, 
                 data_root: str, 
                 split_txt: str, 
                 transform=None,
                 skip_invisible: bool = False,
                 use_2d_noise_aug: bool = True,
                 noise_std_base: float = 0.03,
                 noise_prob: float = 0.5):
        
        self.root = data_root
        self.transform = transform
        self.skip_invisible = skip_invisible
        
        self.use_2d_noise_aug = use_2d_noise_aug
        self.noise_std_base = noise_std_base
        self.noise_prob = noise_prob
        
        self.is_training = 'train' in split_txt
        
        with open(split_txt) as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]
        
        j = lambda *p: os.path.join(data_root, *p)
        self.paths = dict(
            joints_2d=j("annotations", "joints_2d"),
            joints_3d=j("annotations", "joints_3d"),
            rot_mats=j("annotations", "rot_mats"),
            visibility=j("annotations", "visibility"),
            K=j("annotations", "K"),
            R=j("annotations", "R"),
            t=j("annotations", "t"),
            res=j("annotations", "res"),
            rgb=j("raw", "rgb")
        )
    
    def __len__(self) -> int:
        return len(self.ids)


    def add_2d_noise_augmentation(self, joints_2d, visibility_mask, confidence_weights=None, resolution=None):
        """Resolution-aware noise augmentation for PIXEL coordinates"""
        if not self.use_2d_noise_aug or not self.is_training:
            if confidence_weights is not None:
                return joints_2d, confidence_weights
            return joints_2d
        
        if resolution is not None:
            if torch.is_tensor(resolution):
                width = resolution[0].item()
                height = resolution[1].item()
            else:
                width = resolution[0]
                height = resolution[1]
        else:
            width = height = 1920
        
        single_sample = joints_2d.dim() == 2
        if single_sample:
            joints_2d = joints_2d.unsqueeze(0)
            if confidence_weights is not None:
                confidence_weights = confidence_weights.unsqueeze(0)
            if visibility_mask is not None:
                visibility_mask = visibility_mask.unsqueeze(0)
        
        batch_size = joints_2d.shape[0]
        augmented = joints_2d.clone()
        aug_confidence = confidence_weights.clone() if confidence_weights is not None else None
        
        base_resolution = 1920.0
        resolution_scale = max(width, height) / base_resolution
        
        # Minimal dropout
        if torch.rand(1).item() < 0.1:
            dropout_mask = torch.ones(batch_size, 24, dtype=torch.bool)
            
            dropout_probs = torch.ones(24) * 0.005
            dropout_probs[22:24] = 0.02
            dropout_probs[10:12] = 0.015
            
            for b in range(batch_size):
                for j in range(24):
                    if torch.rand(1).item() < dropout_probs[j]:
                        dropout_mask[b, j] = False
            
            for b in range(batch_size):
                for j in range(24):
                    if not dropout_mask[b, j]:
                        noise_scale = 5 * resolution_scale
                        if j == 22:
                            augmented[b, j] = augmented[b, 20] + torch.randn(2) * noise_scale
                        elif j == 23:
                            augmented[b, j] = augmented[b, 21] + torch.randn(2) * noise_scale
                        elif j == 10:
                            augmented[b, j] = augmented[b, 7] + torch.randn(2) * noise_scale
                        elif j == 11:
                            augmented[b, j] = augmented[b, 8] + torch.randn(2) * noise_scale
                        else:
                            augmented[b, j] += torch.randn(2) * (10 * resolution_scale)
            
            if aug_confidence is not None:
                aug_confidence = aug_confidence * dropout_mask.float()
                aug_confidence = torch.clamp(aug_confidence, 0.1, 1.0)
        
        # Gaussian noise
        if torch.rand(1).item() < self.noise_prob:
            noise_pixels = self.noise_std_base * 100 * resolution_scale
            noise = torch.randn_like(augmented) * noise_pixels
            
            noise_scale = torch.ones(24, device=augmented.device)
            noise_scale[0:10] = 1.3
            noise_scale[15] = 1.3
            noise_scale[22:24] = 0.9
            noise_scale[10:12] = 0.9
            
            noise = noise * noise_scale.unsqueeze(0).unsqueeze(-1)
            augmented = augmented + noise
        
        # Detection shifts
        if torch.rand(1).item() < 0.2:
            for b in range(batch_size):
                if torch.rand(1).item() < 0.3:
                    shift = torch.randn(1, 2, device=augmented.device) * (3 * resolution_scale)
                    augmented[b] += shift
                
                if torch.rand(1).item() < 0.2:
                    arm_joints = [16, 18, 20, 22] if torch.rand(1).item() < 0.5 else [17, 19, 21, 23]
                    drift = torch.randn(1, 2, device=augmented.device) * (4 * resolution_scale)
                    for j in arm_joints:
                        augmented[b, j] += drift[0]
        
        augmented[:, :, 0] = torch.clamp(augmented[:, :, 0], 0, width)
        augmented[:, :, 1] = torch.clamp(augmented[:, :, 1], 0, height)
        
        if single_sample:
            augmented = augmented.squeeze(0)
            if aug_confidence is not None:
                aug_confidence = aug_confidence.squeeze(0)
        
        if confidence_weights is None:
            return augmented
        else:
            return augmented, aug_confidence

    def create_synthetic_confidence(self, visibility_mask):
        """Create synthetic confidence values matching real 3DPW data patterns"""
        base_confidence = torch.ones(24, dtype=torch.float32)
        
        base_confidence[15] = 0.894
        base_confidence[12] = 0.785
        base_confidence[7] = 0.805
        base_confidence[8] = 0.796
        base_confidence[4] = 0.789
        base_confidence[5] = 0.789
        
        base_confidence[16] = 0.760
        base_confidence[17] = 0.760
        base_confidence[18] = 0.739
        base_confidence[19] = 0.739
        base_confidence[1] = 0.720
        base_confidence[2] = 0.720
        
        base_confidence[0] = 0.650
        base_confidence[3] = 0.580
        base_confidence[6] = 0.540
        base_confidence[20] = 0.519
        base_confidence[21] = 0.496
        
        base_confidence[9] = 0.496
        base_confidence[10] = 0.450
        base_confidence[11] = 0.450
        
        base_confidence[13] = 0.396
        base_confidence[14] = 0.396
        base_confidence[22] = 0.311
        base_confidence[23] = 0.298
        
        if self.is_training:
            sample_quality = torch.randn(1).item()
            
            if sample_quality < -1.5:
                global_scale = 0.7
                base_var = 0.15
            elif sample_quality < -0.5:
                global_scale = 0.85
                base_var = 0.10
            elif sample_quality > 1.5:
                global_scale = 1.15
                base_var = 0.03
            elif sample_quality > 0.5:
                global_scale = 1.05
                base_var = 0.05
            else:
                global_scale = 1.0
                base_var = 0.07
            
            confidence = base_confidence * global_scale
            
            variation = torch.randn(24) * base_var
            
            variation[22:24] *= 2.0
            variation[20:22] *= 1.8
            variation[10:12] *= 1.5
            variation[13:15] *= 1.3
            
            if torch.rand(1).item() < 0.20:
                issue_type = torch.rand(1).item()
                if issue_type < 0.33:
                    variation[20:24] -= 0.15
                    variation[10:12] -= 0.10
                elif issue_type < 0.66:
                    variation *= 1.5
                else:
                    num_affected = torch.randint(2, 6, (1,)).item()
                    affected_joints = torch.randperm(24)[:num_affected]
                    variation[affected_joints] -= 0.20
            
            confidence = confidence + variation
            confidence = torch.clamp(confidence, 0.1, 0.95)
        else:
            variation = torch.randn(24) * 0.02
            confidence = torch.clamp(base_confidence + variation, 0.1, 0.95)
        
        confidence = confidence * visibility_mask.float()
        
        return confidence

    def load_sample(self, idx):
        """Load a single sample with comprehensive augmentation"""
        did = self.ids[idx]
        load = lambda key: np.load(os.path.join(self.paths[key], f"{did}.npy"))
        
        joints_2d = load("joints_2d")
        joints_3d = load("joints_3d")
        rot_mats = load("rot_mats")
        visibility = load("visibility")
        K = load("K")
        R = load("R")
        t = load("t")
        res = load("res")
        
        joints_2d_tensor = torch.tensor(joints_2d, dtype=torch.float32)
        joints_3d_tensor = torch.tensor(joints_3d, dtype=torch.float32)
        rot_mats_tensor = torch.tensor(rot_mats, dtype=torch.float32)
        visibility_mask = torch.tensor(visibility, dtype=torch.bool)
        
        K_tensor = torch.tensor(K, dtype=torch.float32)
        R_tensor = torch.tensor(R, dtype=torch.float32)
        t_tensor = torch.tensor(t, dtype=torch.float32)
        resolution = torch.tensor(res, dtype=torch.float32)
        
        root_translation = joints_3d_tensor[0].clone()
        joints_3d_centered = joints_3d_tensor - root_translation
        rot_6d = rot_matrix_to_6d(rot_mats_tensor)
        
        confidence_weights = self.create_synthetic_confidence(visibility_mask)
        
        result = self.add_2d_noise_augmentation(
            joints_2d_tensor, visibility_mask, confidence_weights, resolution
        )
        if isinstance(result, tuple):
            joints_2d_augmented, confidence_weights = result
        else:
            joints_2d_augmented = result
        
        sample = {
            "joints_2d": joints_2d_augmented,
            "joints_3d": joints_3d_tensor,
            "joints_3d_centered": joints_3d_centered,
            "joints_3d_world": joints_3d_tensor.clone(),
            "rot_mats": rot_mats_tensor,
            "rot_6d": rot_6d,
            "data_type": "synthetic",
            "frame_id": did,
            "K": K_tensor,
            "R": R_tensor,
            "t": t_tensor,
            "resolution": resolution,
            "root_translation": root_translation,
            "visibility_mask": visibility_mask,
            "confidence": confidence_weights,
            "confidence_weights": confidence_weights
        }
        
        return sample
        
    def __getitem__(self, idx: int) -> Dict:
        sample = self.load_sample(idx)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class RealPoseAdapter(Dataset):
    """Adapter for real 3DPW data with COCO-WholeBody keypoints"""
    
    def __init__(self, 
                 data_root: str, 
                 split: str = 'validation',
                 transform=None,
                 confidence_threshold: float = 0.3,
                 confidence_mode: str = 'soft'):
        
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.confidence_threshold = confidence_threshold
        self.confidence_mode = confidence_mode
        self.samples = []
        self.is_training = (split == 'train')
        
        self.total_samples = 0
        self.detected_samples = 0
        self.detection_stats = {
            'total_frames': 0,
            'detected_frames': 0,
            'total_actors': 0,
            'detected_actors': 0,
            'sequences': {}
        }
        
        detections_dir = os.path.join(data_root, "3DPW_processed", "detections")
        
        if not os.path.exists(detections_dir):
            raise FileNotFoundError(f"3DPW processed detections not found: {detections_dir}")
        
        for filename in sorted(os.listdir(detections_dir)):
            if not filename.endswith('_detections.pkl'):
                continue
            
            file_split = filename.split('_')[0]
            if file_split != split:
                continue
            
            filepath = os.path.join(detections_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    seq_data = pickle.load(f)
                
                for frame_idx, frame_data in seq_data['detections'].items():
                    for actor_idx, actor_data in frame_data.items():
                        if actor_data['joints_3d_centered'] is not None:
                            self.total_samples += 1
                            
                            if actor_data['matched'] and actor_data['keypoints'] is not None:
                                self.samples.append({
                                    'sequence_name': seq_data['sequence_name'],
                                    'split': seq_data['split'], 
                                    'frame_idx': frame_idx,
                                    'actor_idx': actor_idx,
                                    'data': actor_data
                                })
                                self.detected_samples += 1
            
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue
        
        self.detection_stats['total_actors'] = self.total_samples
        self.detection_stats['detected_actors'] = self.detected_samples
        self.detection_stats['detection_rate'] = (
            self.detected_samples / self.total_samples * 100 if self.total_samples > 0 else 0
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_detection_stats(self):
        return self.detection_stats
    
    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        data = sample_info['data']
        
        joints_2d_coco_wb = torch.tensor(data['keypoints'], dtype=torch.float32)
        scores_coco_wb = torch.tensor(data['scores'], dtype=torch.float32)
        
        joints_2d_smpl_np, confidence_smpl_np = coco_wholebody_to_smpl_with_confidence(
            joints_2d_coco_wb.numpy(),
            scores_coco_wb.numpy()
        )
        
        joints_2d_smpl = torch.tensor(joints_2d_smpl_np, dtype=torch.float32)
        confidence_smpl = torch.tensor(confidence_smpl_np, dtype=torch.float32)
        
        if self.confidence_mode == 'soft':
            visibility_mask = confidence_smpl > self.confidence_threshold
            confidence_weights = confidence_smpl
        elif self.confidence_mode == 'hard':
            visibility_mask = confidence_smpl > self.confidence_threshold
            confidence_weights = visibility_mask.float()
        else:
            visibility_mask = torch.ones(24, dtype=torch.bool)
            confidence_weights = confidence_smpl
        
        if self.is_training:
            noise = torch.randn_like(confidence_weights) * 0.1
            
            noise[20:24] *= 2.0
            noise[10:12] *= 1.5
            
            confidence_weights = torch.clamp(confidence_weights + noise, 0.05, 1.0)
        
        joints_3d_centered = torch.tensor(data['joints_3d_centered'], dtype=torch.float32)
        rot_mats = torch.tensor(data['rot_mats'], dtype=torch.float32)
        rot_6d = torch.tensor(data['rot_6d'], dtype=torch.float32)
        
        K = torch.tensor(data['K'], dtype=torch.float32)
        R = torch.tensor(data['R'], dtype=torch.float32)
        t = torch.tensor(data['t'], dtype=torch.float32)
        resolution = torch.tensor(data['resolution'], dtype=torch.float32)
        root_translation = torch.tensor(data['root_translation'], dtype=torch.float32)
        
        sample = {
            "joints_2d": joints_2d_smpl,
            "joints_3d": joints_3d_centered.clone(),
            "joints_3d_centered": joints_3d_centered,
            "joints_3d_world": joints_3d_centered + root_translation,
            "rot_mats": rot_mats,
            "rot_6d": rot_6d,
            "confidence": confidence_smpl,
            "visibility_mask": visibility_mask,
            "confidence_weights": confidence_weights,
            "data_type": "real",
            "frame_id": f"{sample_info['sequence_name']}_{sample_info['frame_idx']}_{sample_info['actor_idx']}",
            "K": K,
            "R": R,
            "t": t,
            "resolution": resolution,
            "root_translation": root_translation
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_dataset(data_root: str,
                   dataset_type: str,
                   split: str = 'train',
                   transform=None,
                   confidence_threshold: float = 0.3,
                   confidence_mode: str = 'soft',
                   skip_invisible: bool = False,
                   use_2d_noise_aug: bool = True,
                   noise_std_base: float = 0.03,
                   noise_prob: float = 0.5,
                   **kwargs) -> Union[SyntheticPoseAdapter, RealPoseAdapter]:
    """Create dataset with appropriate settings"""
    
    if dataset_type == 'synthetic':
        split_txt = os.path.join(data_root, "splits", f"{split}.txt")
        return SyntheticPoseAdapter(
            data_root=data_root,
            split_txt=split_txt,
            transform=transform,
            skip_invisible=skip_invisible,
            use_2d_noise_aug=use_2d_noise_aug,
            noise_std_base=noise_std_base,
            noise_prob=noise_prob
        )
    
    elif dataset_type == 'real':
        real_split = {'train': 'train', 'val': 'validation', 'test': 'test'}.get(split, split)
        return RealPoseAdapter(
            data_root=data_root,
            split=real_split,
            transform=transform,
            confidence_threshold=confidence_threshold,
            confidence_mode=confidence_mode
        )
    
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Choose 'synthetic' or 'real'")