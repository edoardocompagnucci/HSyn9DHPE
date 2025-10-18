import os
import glob
import random

PREFIX_PERCENTAGES = {
    "AMASS": 1.0,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
ANN_DIR = os.path.join(DATA_ROOT, "annotations")
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

files = glob.glob(os.path.join(ANN_DIR, "*.npz"))
all_frame_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]

print(f"Found {len(files)} sequences")

frame_ids_by_prefix = {}
for prefix in PREFIX_PERCENTAGES.keys():
    frame_ids_by_prefix[prefix] = [fid for fid in all_frame_ids if fid.startswith(prefix)]

selected_frame_ids = []
random.seed(1257)

for prefix, percentage in PREFIX_PERCENTAGES.items():
    prefix_ids = frame_ids_by_prefix[prefix]
    if not prefix_ids:
        continue

    random.shuffle(prefix_ids)

    num_to_select = int(len(prefix_ids) * percentage)
    selected_ids = prefix_ids[:num_to_select]
    selected_frame_ids.extend(selected_ids)

if not selected_frame_ids:
    raise RuntimeError(f"No .npz files selected from {ANN_DIR}")

random.shuffle(selected_frame_ids)

train_path = os.path.join(SPLITS_DIR, "train.txt")

with open(train_path, "w") as f:
    f.write("\n".join(selected_frame_ids) + "\n")