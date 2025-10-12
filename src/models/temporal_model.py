"""
Minimal temporal 3D pose model
Based on VideoPose3D architecture: simple temporal convolutions

Design:
1. Reuse GraphFormer encoder (spatial reasoning per frame)
2. Add dilated temporal convolutions (capture motion patterns)
3. Simple, proven, easy to train
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.graphformer import GraphFormerPose


class TemporalConv(nn.Module):
    """
    Temporal convolution block with dilation
    Based on VideoPose3D and other temporal pose papers
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()

        # Padding to preserve temporal dimension
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] tensor
        Returns:
            [B, C_out, T] tensor
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x


class TemporalPoseModel(nn.Module):
    """
    Temporal 3D pose estimation model

    Architecture:
        Input: [B, T, J, 2] 2D keypoints
          ↓
        Frame-wise GraphFormer encoder → [B, T, J, D] features
          ↓
        Reshape → [B, J*D, T]
          ↓
        Temporal convolutions (1-2 blocks)
          ↓
        Reshape → [B, T, J, D]
          ↓
        Frame-wise prediction heads → [B, T, J*3] positions + [B, T, J, 6] rotations
    """

    def __init__(self,
                 num_joints: int = 24,
                 dim: int = 384,
                 depth: int = 8,
                 heads: int = 12,
                 ffn_dim: int = 1536,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 2,
                 temporal_kernel_size: int = 3,
                 use_dilation: bool = True):
        super().__init__()

        self.num_joints = num_joints
        self.dim = dim

        # Spatial encoder (reuse proven GraphFormer)
        self.spatial_encoder = GraphFormerPose(
            num_joints=num_joints,
            dim=dim,
            depth=depth,
            heads=heads,
            ffn_dim=ffn_dim,
            dropout=dropout
        )

        # Freeze spatial encoder initially (optional - can be unfrozen later)
        # This allows training temporal components first
        self.freeze_spatial = False  # Set to True to train only temporal

        # Temporal processing
        # Process all joints together to capture inter-joint temporal patterns
        temporal_in_channels = num_joints * dim

        self.temporal_blocks = nn.ModuleList()

        for i in range(num_temporal_layers):
            # Use increasing dilation to capture patterns at multiple scales
            dilation = 2 ** i if use_dilation else 1

            self.temporal_blocks.append(
                TemporalConv(
                    in_channels=temporal_in_channels,
                    out_channels=temporal_in_channels,
                    kernel_size=temporal_kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        # Output heads (same as GraphFormer)
        self.position_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 3)
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 6)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights carefully"""
        # Temporal convolutions: small init to avoid disrupting spatial features
        for module in self.temporal_blocks:
            nn.init.kaiming_normal_(module.conv.weight, mode='fan_out', nonlinearity='relu')
            # Scale down to be conservative
            module.conv.weight.data *= 0.1

        # Output heads: same as GraphFormer
        for module in [self.position_head, self.rotation_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Special init for position head (same as GraphFormer)
        with torch.no_grad():
            self.position_head[-1].weight.data.normal_(0, 0.1)
            self.position_head[-1].bias.data.uniform_(-0.3, 0.3)

    def forward(self, keypoints_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            keypoints_2d: [B, T, J, 2] tensor

        Returns:
            positions_3d: [B, T, J*3] tensor (centered at pelvis)
            rotations_6d: [B, T, J, 6] tensor
        """
        B, T, J, _ = keypoints_2d.shape

        # 1. Extract spatial features (batch all frames together for speed!)
        # Reshape [B, T, J, 2] -> [B*T, J, 2]
        keypoints_2d_flat = keypoints_2d.reshape(B * T, J, 2)

        if self.freeze_spatial:
            self.spatial_encoder.eval()
            with torch.no_grad():
                features_flat = self.spatial_encoder.get_features(keypoints_2d_flat)  # [B*T, J, D]
        else:
            features_flat = self.spatial_encoder.get_features(keypoints_2d_flat)  # [B*T, J, D]

        # Reshape back to [B, T, J, D]
        features = features_flat.reshape(B, T, J, self.dim)

        # 2. Temporal processing
        # Reshape for 1D convolution: [B, J*D, T]
        feat_reshaped = features.reshape(B, T, -1).transpose(1, 2)

        # Apply temporal convolutions
        temp_feat = feat_reshaped
        for block in self.temporal_blocks:
            # Residual connection
            temp_feat = temp_feat + block(temp_feat)

        # Reshape back: [B, T, J, D]
        features = temp_feat.transpose(1, 2).reshape(B, T, J, self.dim)

        # 3. Generate outputs per frame
        positions_3d = self.position_head(features)  # [B, T, J, 3]
        rotations_6d = self.rotation_head(features)  # [B, T, J, 6]

        # Center positions (pelvis = 0)
        positions_3d = positions_3d - positions_3d[:, :, 0:1, :]

        # Reshape positions to [B, T, J*3]
        positions_3d = positions_3d.reshape(B, T, -1)

        return positions_3d, rotations_6d

    def freeze_spatial_encoder(self):
        """Freeze spatial encoder for training only temporal components"""
        self.freeze_spatial = True
        for param in self.spatial_encoder.parameters():
            param.requires_grad = False
        print("Spatial encoder frozen - training only temporal components")

    def unfreeze_spatial_encoder(self):
        """Unfreeze for end-to-end training"""
        self.freeze_spatial = False
        for param in self.spatial_encoder.parameters():
            param.requires_grad = True
        print("Spatial encoder unfrozen - end-to-end training")

    def load_spatial_encoder(self, checkpoint_path: str):
        """Load pretrained GraphFormer weights"""
        print(f"Loading pretrained spatial encoder from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Load into spatial encoder
        missing, unexpected = self.spatial_encoder.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"  Missing keys (expected for output heads): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        print("  Pretrained weights loaded successfully!")

        # Print validation metrics from checkpoint
        if 'val_pa_mpjpe' in checkpoint:
            val_pa_mpjpe = checkpoint['val_pa_mpjpe'] * 1000
            print(f"  Pretrained model PA-MPJPE: {val_pa_mpjpe:.1f}mm")


# Add method to GraphFormer to extract features
def graphformer_get_features(self, keypoints_2d: torch.Tensor) -> torch.Tensor:
    """
    Extract features before output heads

    Args:
        keypoints_2d: [B, J, 2]
    Returns:
        features: [B, J, D]
    """
    B = keypoints_2d.shape[0]

    # Input projection
    x = self.input_proj(keypoints_2d) + self.joint_embed

    # Process through encoder layers (attribute is 'layers' in GraphFormer)
    for layer in self.layers:
        x = layer(x, self.laplacian)

    return x


# Monkey-patch GraphFormer to add feature extraction
GraphFormerPose.get_features = graphformer_get_features
