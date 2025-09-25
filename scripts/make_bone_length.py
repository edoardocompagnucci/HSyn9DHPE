import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm

# Add src to path to import skeleton
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.skeleton import SMPL_SKELETON, SMPL_JOINT_NAMES

np.random.seed(42)

edges = SMPL_SKELETON

# Get NPZ files from annotations directory
npz_files = glob("data/annotations/*.npz")

if not npz_files:
    print("No NPZ files found in data/annotations/")
    sys.exit(1)

# Sample files and frames
n_files = min(100, len(npz_files))  # Use up to 100 files
n_frames_per_file = 50  # Use 50 frames from each file

sampled_files = np.random.choice(npz_files, size=n_files, replace=False)

print(f"Processing {n_files} NPZ files out of {len(npz_files)} total files")
print(f"Using {n_frames_per_file} frames per file")

bone_lengths_all = []

for npz_path in tqdm(sampled_files, desc="Processing files"):
    try:
        # Load NPZ file
        data = np.load(npz_path)
        joints_3d = data['joints_3d']  # [num_frames, 24, 3]
        
        # Sample frames from this file
        num_frames = joints_3d.shape[0]
        frames_to_use = min(n_frames_per_file, num_frames)
        
        # Take evenly spaced frames
        frame_indices = np.linspace(0, num_frames-1, frames_to_use, dtype=int)
        
        for frame_idx in frame_indices:
            joints = joints_3d[frame_idx]  # [24, 3]
            
            # Compute bone lengths for this frame
            lengths = []
            for (parent, child) in edges:
                bone_vec = joints[child] - joints[parent]
                length = np.linalg.norm(bone_vec)
                lengths.append(length)
            
            bone_lengths_all.append(lengths)
            
    except Exception as e:
        print(f"Error processing {os.path.basename(npz_path)}: {e}")
        continue

bone_lengths_all = np.array(bone_lengths_all)

# Use median instead of mean for robustness
bone_lengths = np.median(bone_lengths_all, axis=0)
bone_lengths_std = np.std(bone_lengths_all, axis=0)

# Create 24x24 matrix format for compatibility with loss function
bone_matrix = np.zeros((24, 24), dtype=np.float32)
for idx, (parent, child) in enumerate(edges):
    bone_matrix[parent, child] = bone_lengths[idx]
    bone_matrix[child, parent] = bone_lengths[idx]  # Symmetric

# Save both formats
os.makedirs("data/meta", exist_ok=True)
np.save("data/meta/bone_lengths.npy", bone_lengths)  # Vector format (23 bones)
np.save("data/meta/bone_length.npy", bone_matrix)    # Matrix format (24x24)

print(f"\n✅ Saved bone lengths from {len(bone_lengths_all)} total frames")
print(f"   - bone_lengths.npy: shape {bone_lengths.shape} (vector of {len(edges)} bones)")
print(f"   - bone_length.npy: shape {bone_matrix.shape} (24x24 matrix)")

# Print bone length statistics
print("\n📊 Bone Length Statistics (in meters):")
print("-" * 50)
for idx, (parent, child) in enumerate(edges):
    parent_name = SMPL_JOINT_NAMES[parent]
    child_name = SMPL_JOINT_NAMES[child]
    print(f"{parent_name:12} -> {child_name:12}: {bone_lengths[idx]:.4f}m (±{bone_lengths_std[idx]:.4f})")

# Summary statistics
print("\n📏 Summary:")
print(f"   Average bone length: {np.mean(bone_lengths):.4f}m")
print(f"   Shortest bone: {np.min(bone_lengths):.4f}m")
print(f"   Longest bone: {np.max(bone_lengths):.4f}m")