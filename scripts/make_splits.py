import os
import glob
import random

PREFIX_PERCENTAGES = {
    "AMASS": 1.0,  # 50% of AMASS data (~3300 sequences) - balance between quality and training time
    # Add other prefixes as needed
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
ANN_DIR = os.path.join(DATA_ROOT, "annotations")  # Now points to annotations root
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

# Look for NPZ files instead of NPY
files = glob.glob(os.path.join(ANN_DIR, "*.npz"))
all_frame_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]

print(f"Found {len(files)} NPZ files in {ANN_DIR}")

frame_ids_by_prefix = {}
for prefix in PREFIX_PERCENTAGES.keys():
    frame_ids_by_prefix[prefix] = [fid for fid in all_frame_ids if fid.startswith(prefix)]
    print(f"Found {len(frame_ids_by_prefix[prefix])} files with prefix '{prefix}'")

selected_frame_ids = []
random.seed(1257)

for prefix, percentage in PREFIX_PERCENTAGES.items():
    prefix_ids = frame_ids_by_prefix[prefix]
    if not prefix_ids:
        print(f"Warning: No files found with prefix '{prefix}'")
        continue
    
    random.shuffle(prefix_ids)
    
    num_to_select = int(len(prefix_ids) * percentage)
    selected_ids = prefix_ids[:num_to_select]
    selected_frame_ids.extend(selected_ids)
    
    print(f"Selected {num_to_select} ({percentage*100:.1f}%) out of {len(prefix_ids)} '{prefix}' files")

if not selected_frame_ids:
    raise RuntimeError(f"No .npz files selected from {ANN_DIR}")

random.shuffle(selected_frame_ids)

train_path = os.path.join(SPLITS_DIR, "train.txt")

with open(train_path, "w") as f:
    f.write("\n".join(selected_frame_ids) + "\n")

print(f"\nTotal selected: {len(selected_frame_ids)} samples")
print(f"Saved {len(selected_frame_ids)} train IDs → {train_path}")

print("\nSummary by prefix:")
for prefix in PREFIX_PERCENTAGES.keys():
    count = sum(1 for id in selected_frame_ids if id.startswith(prefix))
    print(f"  {prefix}: {count} samples")