import os
import sys
import numpy as np
from glob import glob

# Add src to path to import skeleton
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.skeleton import SMPL_SKELETON

np.random.seed(42)

edges = SMPL_SKELETON

j3d_files = glob("data/annotations/joints_3d/*.npy")

n_samples = min(10000, len(j3d_files))
sampled_files = np.random.choice(j3d_files, size=n_samples, replace=False)

print(f"Processing {n_samples} files out of {len(j3d_files)} total files")

bone_lengths_all = []

for f in sampled_files:
    joints = np.load(f)
    lengths = [np.linalg.norm(joints[a] - joints[b]) for (a, b) in edges]
    bone_lengths_all.append(lengths)

bone_lengths_all = np.array(bone_lengths_all)
bone_lengths = bone_lengths_all.mean(axis=0)

os.makedirs("data/meta", exist_ok=True)
np.save("data/meta/bone_lengths.npy", bone_lengths)

print(f"Saved bone_lengths.npy with {len(bone_lengths)} bones.")