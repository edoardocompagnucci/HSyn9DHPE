import torch
import torch.nn.functional as F


def rot_matrix_to_6d(rot_matrices):
    return torch.cat([rot_matrices[..., :, 0], rot_matrices[..., :, 1]], dim=-1)


def rot_6d_to_matrix(rot_6d):
    batch_shape = rot_6d.shape[:-1]
    rot_6d_flat = rot_6d.reshape(-1, 6)

    a1 = rot_6d_flat[:, :3]
    a2 = rot_6d_flat[:, 3:]

    b1 = F.normalize(a1, dim=1, eps=1e-8)

    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1, eps=1e-8)

    b3 = torch.cross(b1, b2, dim=1)

    rot_matrix = torch.stack([b1, b2, b3], dim=-1)
    return rot_matrix.reshape(*batch_shape, 3, 3)
