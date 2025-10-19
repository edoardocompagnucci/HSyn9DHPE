import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from utils.skeleton import SMPL_SKELETON


class GraphMLP(nn.Module):
    def __init__(self, dim: int, order: int = 2, dropout: float = 0.1):
        super().__init__()
        self.order = order
        self.theta = nn.Parameter(torch.randn(order, dim, dim) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        out = x
        Tx = x
        for k in range(self.order):
            Tx = laplacian @ Tx
            out = out + Tx @ self.theta[k]
        return self.proj(out)


class PoseEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.graph_mlp = GraphMLP(dim, order=3, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        y, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.drop(y))

        y = self.graph_mlp(x, laplacian)
        x = self.norm2(x + self.drop(y))

        y = self.ffn(x)
        x = self.norm3(x + self.drop(y))
        return x


class GraphFormerPose(nn.Module):
    """
    Graph Transformer for 3D pose and orientation estimation
    2D keypoints (B, 24, 2) → 3D joint pos (B, 24*3) + 6D joint rot (B, 24, 6)
    """

    _EDGES: List[Tuple[int, int]] = SMPL_SKELETON

    def __init__(
        self,
        num_joints: int = 24,
        in_feats: int = 2,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if ffn_dim is None:
            ffn_dim = 4 * dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_feats, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim // 2, dim),
            nn.LayerNorm(dim)
        )

        self.joint_embed = nn.Parameter(torch.randn(num_joints, dim) * 0.02)

        self.register_buffer(
            "laplacian",
            self._build_laplacian(num_joints),
            persistent=False
        )

        self.layers = nn.ModuleList([
            PoseEncoderLayer(dim, heads, ffn_dim, dropout)
            for _ in range(depth)
        ])

        self.position_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 3)
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 6)
        )

        self._init_weights()
        with torch.no_grad():
            self.position_head[-1].weight.data.normal_(0, 0.1)
            self.position_head[-1].bias.data.uniform_(-0.3, 0.3)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def _build_laplacian(self, J: int) -> torch.Tensor:
        A = torch.eye(J)
        
        for i, j in self._EDGES:
            if i < J and j < J:
                A[i, j] = A[j, i] = 1.0
        
        deg = A.sum(1)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        return deg_inv_sqrt @ A @ deg_inv_sqrt

    def forward(self, keypoints_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = keypoints_2d.shape[0]
        
        x = self.input_proj(keypoints_2d) + self.joint_embed

        for layer in self.layers:
            x = layer(x, self.laplacian)

        positions_3d = self.position_head(x)
        orientations_6d = self.rotation_head(x)

        positions_3d = positions_3d - positions_3d[:, 0:1, :]

        pos3d = positions_3d.view(batch_size, -1)
        rot6d = orientations_6d

        return pos3d, rot6d