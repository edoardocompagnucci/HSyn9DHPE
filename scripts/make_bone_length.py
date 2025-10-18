import os
import sys
import numpy as np
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.skeleton import SMPL_SKELETON, SMPL_JOINT_NAMES

np.random.seed(42)

edges = SMPL_SKELETON

npz_files = glob("data/annotations/*.npz")

if not npz_files:
    sys.exit(1)

n_files = min(100, len(npz_files))
n_frames_per_file = 50

sampled_files = np.random.choice(npz_files, size=n_files, replace=False)

bone_lengths_all = []

for npz_path in sampled_files:
    try:
        data = np.load(npz_path)
        joints_3d = data['joints_3d']

        num_frames = joints_3d.shape[0]
        frames_to_use = min(n_frames_per_file, num_frames)

        frame_indices = np.linspace(0, num_frames-1, frames_to_use, dtype=int)

        for frame_idx in frame_indices:
            joints = joints_3d[frame_idx]

            lengths = []
            for (parent, child) in edges:
                bone_vec = joints[child] - joints[parent]
                length = np.linalg.norm(bone_vec)
                lengths.append(length)

            bone_lengths_all.append(lengths)

    except Exception:
        continue

bone_lengths_all = np.array(bone_lengths_all)

bone_lengths = np.median(bone_lengths_all, axis=0)
bone_lengths_std = np.std(bone_lengths_all, axis=0)

bone_matrix = np.zeros((24, 24), dtype=np.float32)
for idx, (parent, child) in enumerate(edges):
    bone_matrix[parent, child] = bone_lengths[idx]
    bone_matrix[child, parent] = bone_lengths[idx]

os.makedirs("data/meta", exist_ok=True)
np.save("data/meta/bone_lengths.npy", bone_lengths)
np.save("data/meta/bone_length.npy", bone_matrix)