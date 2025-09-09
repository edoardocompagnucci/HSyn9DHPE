import torch
import os
import json

from utils.rotation_utils import rot_6d_to_matrix

_bone_data_loaded = False
_edges = None


def _load_bone_data():
    global _bone_data_loaded, _edges
    
    if _bone_data_loaded:
        return

    from .skeleton import SMPL_SKELETON
    _edges = SMPL_SKELETON
    
    _bone_data_loaded = True


def get_extremity_weights(device='cpu'):
    """Get joint weights that emphasize extremities"""
    weights = torch.ones(24, device=device)
    
    weights[22] = 3.0  # L_Hand
    weights[23] = 3.0  # R_Hand
    weights[20] = 2.5  # L_Wrist
    weights[21] = 2.5  # R_Wrist
    weights[10] = 2.0  # L_Foot
    weights[11] = 2.0  # R_Foot
    weights[7] = 1.5   # L_Ankle
    weights[8] = 1.5   # R_Ankle
    weights[4] = 1.2   # L_Knee
    weights[5] = 1.2   # R_Knee
    weights[18] = 1.3  # L_Elbow
    weights[19] = 1.3  # R_Elbow
    weights[15] = 1.2  # Head
    
    return weights


def position_mse_loss_with_visibility(predicted, target, visibility_mask=None, 
                                     confidence_weights=None, use_extremity_weights=True):
    """MSE loss that handles missing/low-confidence joints with extremity focus"""
    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1] // 3
    
    pred = predicted.reshape(batch_size, num_joints, 3)
    targ = target.reshape(batch_size, num_joints, 3)
    
    if use_extremity_weights:
        joint_weights = get_extremity_weights(pred.device)
    else:
        joint_weights = torch.ones(num_joints, device=pred.device)
    
    squared_error = (pred - targ) ** 2
    joint_mse = squared_error.mean(dim=2)
    joint_mse = joint_mse * joint_weights.unsqueeze(0)
    
    if visibility_mask is not None:
        joint_mse = joint_mse * visibility_mask.float()
        effective_weights = joint_weights.unsqueeze(0) * visibility_mask.float()
        total_weight = effective_weights.sum(dim=1).clamp(min=0.1)
        mse_per_sample = joint_mse.sum(dim=1) / total_weight
        return mse_per_sample.mean()
    
    elif confidence_weights is not None:
        combined_weights = confidence_weights * joint_weights.unsqueeze(0)
        weighted_mse = joint_mse * combined_weights
        total_weight = combined_weights.sum(dim=1).clamp(min=0.1)
        mse_per_sample = weighted_mse.sum(dim=1) / total_weight
        return mse_per_sample.mean()
    
    else:
        total_weight = joint_weights.sum()
        return joint_mse.sum() / (batch_size * total_weight)


def mpjpe_loss_with_visibility(predicted, target, visibility_mask=None, confidence_weights=None):
    """MPJPE that handles missing/low-confidence joints properly"""
    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1] // 3
    
    pred = predicted.reshape(batch_size, num_joints, 3)
    targ = target.reshape(batch_size, num_joints, 3)
    
    joint_errors = torch.sqrt(torch.sum((pred - targ) ** 2, dim=2))
    
    if visibility_mask is not None:
        joint_errors = joint_errors * visibility_mask.float()
        num_visible = visibility_mask.sum(dim=1).clamp(min=1)
        mpjpe_per_sample = joint_errors.sum(dim=1) / num_visible
        return mpjpe_per_sample.mean()
    
    elif confidence_weights is not None:
        weighted_errors = joint_errors * confidence_weights
        total_confidence = confidence_weights.sum(dim=1).clamp(min=0.1)
        mpjpe_per_sample = weighted_errors.sum(dim=1) / total_confidence
        return mpjpe_per_sample.mean()
    
    else:
        return joint_errors.mean()


def pa_mpjpe_with_visibility(predicted, target, visibility_mask=None, confidence_weights=None):
    """Procrustes-aligned MPJPE - aligns predicted pose to target before computing error"""
    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1] // 3
    
    pred = predicted.reshape(batch_size, num_joints, 3).float()
    targ = target.reshape(batch_size, num_joints, 3).float()
    
    if visibility_mask is not None:
        visibility_mask = visibility_mask.float()
    if confidence_weights is not None:
        confidence_weights = confidence_weights.float()
    
    pa_mpjpe_per_sample = []
    
    for i in range(batch_size):
        pred_i = pred[i]
        targ_i = targ[i]
        
        if visibility_mask is not None:
            visible = visibility_mask[i].bool()
            if visible.sum() < 3:
                continue
            pred_i = pred_i[visible]
            targ_i = targ_i[visible]
        elif confidence_weights is not None:
            confident = confidence_weights[i] > 0.5
            if confident.sum() < 3:
                continue
            pred_i = pred_i[confident]
            targ_i = targ_i[confident]
        
        pred_centered = pred_i - pred_i.mean(dim=0, keepdim=True)
        targ_centered = targ_i - targ_i.mean(dim=0, keepdim=True)
        
        scale_pred = torch.sqrt((pred_centered ** 2).sum())
        scale_targ = torch.sqrt((targ_centered ** 2).sum())
        pred_normalized = pred_centered / scale_pred
        targ_normalized = targ_centered / scale_targ
        
        H = pred_normalized.T @ targ_normalized
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        pred_aligned = (pred[i] - pred[i].mean(dim=0, keepdim=True)) @ R.T
        pred_aligned = pred_aligned * (scale_targ / scale_pred)
        pred_aligned = pred_aligned + targ[i].mean(dim=0, keepdim=True)
        
        if visibility_mask is not None:
            errors = torch.sqrt(((pred_aligned - targ[i]) ** 2).sum(dim=1))
            errors = errors * visibility_mask[i]
            mpjpe = errors.sum() / visibility_mask[i].sum().clamp(min=1)
        elif confidence_weights is not None:
            errors = torch.sqrt(((pred_aligned - targ[i]) ** 2).sum(dim=1))
            weighted_errors = errors * confidence_weights[i]
            mpjpe = weighted_errors.sum() / confidence_weights[i].sum().clamp(min=0.1)
        else:
            mpjpe = torch.sqrt(((pred_aligned - targ[i]) ** 2).sum(dim=1)).mean()
        
        pa_mpjpe_per_sample.append(mpjpe)
    
    if len(pa_mpjpe_per_sample) == 0:
        return torch.tensor(0.0, device=predicted.device)
    
    return torch.stack(pa_mpjpe_per_sample).mean()


def pck_3d_with_visibility(predicted, target, thresholds=[50, 100, 150], visibility_mask=None, confidence_weights=None):
    """3D PCK (Percentage of Correct Keypoints) at various thresholds (in mm)"""
    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1] // 3
    
    pred = predicted.reshape(batch_size, num_joints, 3).float()
    targ = target.reshape(batch_size, num_joints, 3).float()
    
    if visibility_mask is not None:
        visibility_mask = visibility_mask.float()
    if confidence_weights is not None:
        confidence_weights = confidence_weights.float()
    
    pred_mm = pred * 1000
    targ_mm = targ * 1000
    
    joint_errors = torch.sqrt(torch.sum((pred_mm - targ_mm) ** 2, dim=2))
    
    pck_results = {}
    
    for threshold in thresholds:
        correct = (joint_errors < threshold).float()
        
        if visibility_mask is not None:
            correct = correct * visibility_mask
            total_visible = visibility_mask.sum()
            pck = correct.sum() / total_visible.clamp(min=1)
        elif confidence_weights is not None:
            weighted_correct = correct * confidence_weights
            pck = weighted_correct.sum() / confidence_weights.sum().clamp(min=0.1)
        else:
            pck = correct.mean()
        
        pck_results[f'pck_{threshold}'] = pck.item() * 100
    
    return pck_results


def rotation_error_metric(pred_6d, target_6d, visibility_mask=None):
    """Compute rotation error in degrees for evaluation"""
    pred_6d = pred_6d.float()
    target_6d = target_6d.float()
    
    if visibility_mask is not None:
        visibility_mask = visibility_mask.float()
    
    if pred_6d.dim() == 2:
        batch_size = pred_6d.shape[0]
        num_joints = pred_6d.shape[1] // 6
        pred_6d = pred_6d.reshape(batch_size, num_joints, 6)
        target_6d = target_6d.reshape(batch_size, num_joints, 6)
    else:
        batch_size = pred_6d.shape[0]
        num_joints = pred_6d.shape[1]
    
    pred_rot = rot_6d_to_matrix(pred_6d)
    target_rot = rot_6d_to_matrix(target_6d)
    
    relative_rot = torch.matmul(pred_rot.transpose(-1, -2), target_rot)
    trace = torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(dim=-1)
    trace_clamped = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
    angle_rad = torch.acos(trace_clamped)
    angle_deg = angle_rad * 180 / torch.pi
    
    if visibility_mask is not None:
        angle_deg = angle_deg * visibility_mask
        num_visible = visibility_mask.sum(dim=1).clamp(min=1)
        mean_error_per_sample = angle_deg.sum(dim=1) / num_visible
        mean_error = mean_error_per_sample.mean()
    else:
        mean_error = angle_deg.mean()
    
    per_joint_errors = angle_deg.mean(dim=0)
    
    return mean_error, per_joint_errors


def geodesic_loss(pred_6d, target_6d):
    """Geodesic loss for rotations"""
    if pred_6d.dim() == 2:
        batch_size = pred_6d.shape[0]
        num_joints = pred_6d.shape[1] // 6
        pred_6d = pred_6d.reshape(batch_size, num_joints, 6)
        target_6d = target_6d.reshape(batch_size, num_joints, 6)

    pred_rot = rot_6d_to_matrix(pred_6d)
    target_rot = rot_6d_to_matrix(target_6d)

    relative_rot = torch.matmul(pred_rot.transpose(-1, -2), target_rot)
    trace = torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(dim=-1)

    trace_clamped = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
    geodesic_dist = torch.acos(trace_clamped)

    return geodesic_dist.mean()


def bone_length_consistency_loss(predicted_poses, target_poses):
    """Ensure predicted bone lengths match the GT skeleton's proportions"""
    _load_bone_data()
    
    batch_size = predicted_poses.shape[0]
    
    if predicted_poses.dim() == 2:
        predicted_poses = predicted_poses.reshape(batch_size, 24, 3)
    if target_poses.dim() == 2:
        target_poses = target_poses.reshape(batch_size, 24, 3)
    
    gt_lengths = []
    pred_lengths = []
    
    for parent, child in _edges:
        gt_bone = target_poses[:, child] - target_poses[:, parent]
        gt_length = torch.norm(gt_bone, dim=-1)
        gt_lengths.append(gt_length)
        
        pred_bone = predicted_poses[:, child] - predicted_poses[:, parent]
        pred_length = torch.norm(pred_bone, dim=-1)
        pred_lengths.append(pred_length)
    
    gt_lengths = torch.stack(gt_lengths, dim=1)
    pred_lengths = torch.stack(pred_lengths, dim=1)
    
    ref_bone_idx = 0
    
    gt_ref_length = gt_lengths[:, ref_bone_idx:ref_bone_idx+1] + 1e-6
    pred_ref_length = pred_lengths[:, ref_bone_idx:ref_bone_idx+1] + 1e-6
    
    gt_normalized = gt_lengths / gt_ref_length
    pred_normalized = pred_lengths / pred_ref_length
    
    loss = torch.mean((pred_normalized - gt_normalized) ** 2)
    
    return loss


def projection_loss_with_visibility(pred_3d_centered, gt_2d, camera_params, root_translation, 
                                   visibility_mask=None, confidence_weights=None, use_extremity_weights=True):
    """Projection loss that respects joint visibility/confidence and extremity importance"""
    batch_size = pred_3d_centered.shape[0]
    device = pred_3d_centered.device
    
    if pred_3d_centered.dim() == 2:
        pred_3d_centered = pred_3d_centered.reshape(batch_size, 24, 3)
    if gt_2d.dim() == 2:
        gt_2d = gt_2d.reshape(batch_size, 24, 2)
    
    if use_extremity_weights:
        joint_weights = get_extremity_weights(device)
    else:
        joint_weights = torch.ones(24, device=device)
    
    K = camera_params['K']
    R = camera_params['R']
    t = camera_params['t']
    resolution = camera_params['resolution']
    
    pred_3d_world = pred_3d_centered + root_translation.unsqueeze(1)
    
    projected_2d = torch.zeros(batch_size, 24, 2, device=device)
    
    for i in range(batch_size):
        K_i = K[i]
        R_i = R[i]
        t_i = t[i]
        
        joints_3d_i = pred_3d_world[i]
        cam_pts = (R_i @ joints_3d_i.T) + t_i.unsqueeze(1)
        
        proj = (K_i @ cam_pts).T
        proj_2d = proj[:, :2] / (proj[:, 2:3] + 1e-8)
        projected_2d[i] = proj_2d

    # Get actual resolution for normalization
    width = resolution[:, 0].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
    height = resolution[:, 1].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
    
    # Normalize to [-1, 1] using actual resolution
    proj_norm_x = (projected_2d[:, :, 0:1] / width) * 2.0 - 1.0
    proj_norm_y = (projected_2d[:, :, 1:2] / height) * 2.0 - 1.0
    proj_norm = torch.cat([proj_norm_x, proj_norm_y], dim=2)
    
    squared_error = (proj_norm - gt_2d) ** 2
    joint_error = squared_error.mean(dim=2)
    joint_error = joint_error * joint_weights.unsqueeze(0)
    
    if visibility_mask is not None:
        joint_error = joint_error * visibility_mask.float()
        effective_weights = joint_weights.unsqueeze(0) * visibility_mask.float()
        total_weight = effective_weights.sum(dim=1).clamp(min=0.1)
        loss_per_sample = joint_error.sum(dim=1) / total_weight
        loss = loss_per_sample.mean()
    
    elif confidence_weights is not None:
        combined_weights = confidence_weights * joint_weights.unsqueeze(0)
        weighted_error = joint_error * combined_weights
        total_weight = combined_weights.sum(dim=1).clamp(min=0.1)
        loss_per_sample = weighted_error.sum(dim=1) / total_weight
        loss = loss_per_sample.mean()
    
    else:
        total_weight = joint_weights.sum()
        loss = joint_error.sum() / (batch_size * total_weight)
    
    # Convert gt_2d back to pixel coordinates for error calculation
    gt_2d_x_pixels = (gt_2d[:, :, 0:1] + 1) * width / 2.0
    gt_2d_y_pixels = (gt_2d[:, :, 1:2] + 1) * height / 2.0
    gt_2d_pixels = torch.cat([gt_2d_x_pixels, gt_2d_y_pixels], dim=2)
    
    pixel_error = torch.sqrt(torch.sum((projected_2d - gt_2d_pixels) ** 2, dim=2)).mean()
    
    return loss, pixel_error


def combined_pose_bone_projection_loss_with_visibility(
    pred_dict, target_dict, 
    camera_params=None, root_translation=None,
    visibility_mask=None, confidence_weights=None,
    pos_weight=1.0, rot_weight=0.1, bone_weight=0.05, 
    projection_weight=0.1, use_geodesic=True, use_extremity_weights=True):
    """Combined loss that properly handles missing joints with extremity focus"""
    
    pos_loss = position_mse_loss_with_visibility(
        pred_dict['positions'], 
        target_dict['positions'],
        visibility_mask, 
        confidence_weights,
        use_extremity_weights=use_extremity_weights
    )
    
    mpjpe = mpjpe_loss_with_visibility(
        pred_dict['positions'], 
        target_dict['positions'],
        visibility_mask,
        confidence_weights
    )
    
    if use_geodesic:
        rot_loss = geodesic_loss(pred_dict['rotations'], target_dict['rotations'])
    else:
        rot_loss = torch.nn.functional.mse_loss(pred_dict['rotations'], target_dict['rotations'])
    
    batch_size = pred_dict['positions'].shape[0]
    pred_positions_3d = pred_dict['positions'].reshape(batch_size, 24, 3)
    target_positions_3d = target_dict['positions'].reshape(batch_size, 24, 3)
    
    if bone_weight > 0:
        bone_loss = bone_length_consistency_loss(pred_positions_3d, target_positions_3d)
    else:
        bone_loss = torch.tensor(0.0, device=pred_dict['positions'].device)
    
    if camera_params is not None and root_translation is not None and 'joints_2d' in target_dict:
        proj_loss, pixel_error = projection_loss_with_visibility(
            pred_positions_3d,
            target_dict['joints_2d'],
            camera_params,
            root_translation,
            visibility_mask,
            confidence_weights,
            use_extremity_weights=use_extremity_weights
        )
    else:
        proj_loss = torch.tensor(0.0, device=pred_dict['positions'].device)
        pixel_error = torch.tensor(0.0, device=pred_dict['positions'].device)
    
    total_loss = (pos_weight * pos_loss + 
                  rot_weight * rot_loss + 
                  bone_weight * bone_loss +
                  projection_weight * proj_loss)
    
    return {
        'total': total_loss,
        'position': pos_loss,
        'rotation': rot_loss,
        'bone': bone_loss,
        'projection': proj_loss,
        'mpjpe': mpjpe,
        'pixel_error': pixel_error
    }