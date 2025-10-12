"""
Temporal losses for 3D pose estimation
Includes position accuracy + temporal smoothness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from utils.losses import (
    position_mse_loss_with_visibility,
    geodesic_loss,
    pa_mpjpe_with_visibility
)


def compute_velocity_loss(positions: torch.Tensor,
                         confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Penalize large velocities (encourages slow, smooth motion)

    Args:
        positions: [B, T, J*3] or [B, T, J, 3]
        confidence: [B, T, J] optional confidence weights

    Returns:
        velocity_loss: scalar
    """
    # Compute velocity
    if positions.dim() == 3:
        velocity = positions[:, 1:] - positions[:, :-1]  # [B, T-1, J*3]
    else:
        raise ValueError(f"Expected 3D tensor, got {positions.dim()}D")

    # L2 norm
    loss = torch.mean(velocity ** 2)

    return loss


def compute_acceleration_loss(positions: torch.Tensor,
                              confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Penalize large accelerations (penalizes jitter)

    Args:
        positions: [B, T, J*3] or [B, T, J, 3]
        confidence: [B, T, J] optional confidence weights

    Returns:
        acceleration_loss: scalar
    """
    # Compute velocity
    if positions.dim() == 3:
        velocity = positions[:, 1:] - positions[:, :-1]  # [B, T-1, J*3]
        # Compute acceleration
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, J*3]
    else:
        raise ValueError(f"Expected 3D tensor, got {positions.dim()}D")

    # L2 norm
    loss = torch.mean(acceleration ** 2)

    return loss


def compute_jitter_metric(positions: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
    """
    Standard jitter metric: mean absolute acceleration in mm/s^2
    Lower is better (smoother motion)

    Args:
        positions: [B, T, J*3] or [B, T, J, 3]
        fps: frame rate

    Returns:
        jitter: scalar (mean absolute acceleration)
    """
    if positions.shape[1] < 3:
        return torch.tensor(0.0, device=positions.device)

    # Compute velocity and acceleration
    if positions.dim() == 3:
        B, T, D = positions.shape
        J = D // 3
        pos_reshaped = positions.reshape(B, T, J, 3)
    else:
        pos_reshaped = positions

    velocity = pos_reshaped[:, 1:] - pos_reshaped[:, :-1]  # [B, T-1, J, 3]
    acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, J, 3]

    # Mean absolute acceleration (in m/frame^2)
    jitter = torch.mean(torch.abs(acceleration))

    # Convert to mm/s^2
    jitter = jitter * 1000 * (fps ** 2)

    return jitter


class TemporalLoss(nn.Module):
    """
    Combined loss for temporal 3D pose estimation

    Components:
    1. Position loss (MSE or MPJPE)
    2. Rotation loss (geodesic)
    3. Velocity loss (temporal smoothness)
    4. Acceleration loss (anti-jitter)
    """

    def __init__(self,
                 position_weight: float = 1.0,
                 rotation_weight: float = 0.005,
                 velocity_weight: float = 0.1,
                 acceleration_weight: float = 0.05,
                 use_geodesic: bool = True):
        super().__init__()

        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.use_geodesic = use_geodesic

    def forward(self,
                pred_positions: torch.Tensor,  # [B, T, J*3]
                target_positions: torch.Tensor,  # [B, T, J*3]
                pred_rotations: torch.Tensor,  # [B, T, J, 6]
                target_rotations: torch.Tensor,  # [B, T, J, 6]
                confidence: Optional[torch.Tensor] = None,  # [B, T, J]
                visibility: Optional[torch.Tensor] = None  # [B, T, J]
                ) -> Dict[str, torch.Tensor]:
        """
        Compute temporal loss

        Returns:
            dict with keys: 'total', 'position', 'rotation', 'velocity', 'acceleration', 'jitter'
        """
        B, T, D = pred_positions.shape
        J = D // 3

        losses = {}

        # 1. Position loss (batched across time for speed - mathematically identical)
        # Reshape to [B*T, J*3] for batch processing
        pred_pos_flat = pred_positions.reshape(B * T, D)
        target_pos_flat = target_positions.reshape(B * T, D)
        vis_flat = visibility.reshape(B * T, J) if visibility is not None else None
        conf_flat = confidence.reshape(B * T, J) if confidence is not None else None

        position_loss = position_mse_loss_with_visibility(
            pred_pos_flat,
            target_pos_flat,
            visibility_mask=vis_flat,
            confidence_weights=conf_flat,
            use_extremity_weights=True
        )
        losses['position'] = position_loss

        # 2. Rotation loss (batched across time for speed)
        rotation_loss = torch.tensor(0.0, device=pred_positions.device)
        if self.rotation_weight > 0:
            # Reshape to [B*T, J, 6]
            pred_rot_flat = pred_rotations.reshape(B * T, J, 6)
            target_rot_flat = target_rotations.reshape(B * T, J, 6)

            if self.use_geodesic:
                rotation_loss = geodesic_loss(pred_rot_flat, target_rot_flat)
            else:
                rotation_loss = F.mse_loss(pred_rot_flat, target_rot_flat)

            losses['rotation'] = rotation_loss

        # 3. Velocity loss (temporal smoothness)
        velocity_loss = torch.tensor(0.0, device=pred_positions.device)
        if T > 1 and self.velocity_weight > 0:
            velocity_loss = compute_velocity_loss(pred_positions, confidence)
            losses['velocity'] = velocity_loss

        # 4. Acceleration loss (anti-jitter)
        acceleration_loss = torch.tensor(0.0, device=pred_positions.device)
        if T > 2 and self.acceleration_weight > 0:
            acceleration_loss = compute_acceleration_loss(pred_positions, confidence)
            losses['acceleration'] = acceleration_loss

        # Total loss
        total_loss = (
            self.position_weight * position_loss +
            self.rotation_weight * rotation_loss +
            self.velocity_weight * velocity_loss +
            self.acceleration_weight * acceleration_loss
        )
        losses['total'] = total_loss

        # Monitoring metrics (not used in loss)
        with torch.no_grad():
            # MPJPE for monitoring (batched)
            pred_3d = pred_positions.reshape(B, T, J, 3)
            target_3d = target_positions.reshape(B, T, J, 3)
            mpjpe = torch.mean(torch.norm(pred_3d - target_3d, dim=-1))
            losses['mpjpe'] = mpjpe

            # Jitter metric
            if T > 2:
                losses['jitter'] = compute_jitter_metric(pred_positions, fps=30.0)

        return total_loss, losses


def temporal_pa_mpjpe(pred_positions: torch.Tensor,
                     target_positions: torch.Tensor,
                     visibility: Optional[torch.Tensor] = None,
                     confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    PA-MPJPE for temporal sequences
    NOTE: Each frame gets its own Procrustes alignment!

    Args:
        pred_positions: [B, T, J*3]
        target_positions: [B, T, J*3]
        visibility: [B, T, J] optional
        confidence: [B, T, J] optional

    Returns:
        pa_mpjpe: scalar (averaged across all frames)
    """
    B, T, _ = pred_positions.shape

    pa_mpjpe_per_frame = []
    for t in range(T):
        frame_pa_mpjpe = pa_mpjpe_with_visibility(
            pred_positions[:, t],
            target_positions[:, t],
            visibility_mask=visibility[:, t] if visibility is not None else None,
            confidence_weights=confidence[:, t] if confidence is not None else None
        )
        pa_mpjpe_per_frame.append(frame_pa_mpjpe)

    return torch.stack(pa_mpjpe_per_frame).mean()
