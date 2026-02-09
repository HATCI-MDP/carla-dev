#!/usr/bin/env python
"""
Preprocess RELLIS-3D for SegFormer fine-tuning.

Converts the raw RELLIS-3D download into HuggingFace-compatible format:
  - Remaps non-contiguous label IDs to contiguous 0..N (void → 255)
  - Organizes into train/val/test splits with images/ and labels/ subdirs
  - Writes id2label.json and label2id.json

Usage:
    python prepare_rellis3d.py --rellis-root /path/to/rellis3d --output /path/to/rellis3d_processed

Expected input structure (after unzipping the Google Drive downloads):
    rellis3d/
    ├── Rellis-3D/
    │   ├── 00000/
    │   │   ├── pylon_camera_node/          # RGB images (.jpg)
    │   │   └── pylon_camera_node_label_id/ # Label images (.png, single-channel)
    │   ├── 00001/
    │   │   └── ...
    │   └── ...
    ├── train.lst   (or in a split folder)
    ├── val.lst
    └── test.lst
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# RELLIS-3D original (non-contiguous) label IDs → class names
# From the RELLIS-3D ontology definition
RELLIS_ORIGINAL_ID2NAME = {
    0: "void",
    1: "dirt",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "object",
    10: "asphalt",
    12: "building",
    15: "log",
    17: "person",
    18: "fence",
    19: "bush",
    23: "concrete",
    27: "barrier",
    29: "puddle",
    30: "mud",
    31: "rubble",
    33: "sign",
    34: "rock",
}

# Remap to contiguous IDs (void=0 maps to 255 ignore index; rest become 0..N-1)
# Build the mapping: skip void (original 0), assign contiguous IDs to the rest
_non_void = {k: v for k, v in sorted(RELLIS_ORIGINAL_ID2NAME.items()) if k != 0}
CONTIGUOUS_ID2NAME = {i: name for i, (_, name) in enumerate(_non_void.items())}
CONTIGUOUS_NAME2ID = {v: k for k, v in CONTIGUOUS_ID2NAME.items()}

# Original ID → contiguous ID (void → 255)
REMAP = {0: 255}  # void → ignore
for new_id, (orig_id, _) in enumerate(sorted(_non_void.items())):
    REMAP[orig_id] = new_id

NUM_CLASSES = len(CONTIGUOUS_ID2NAME)


def remap_label(label_path, output_path):
    """Load a RELLIS-3D label PNG, remap IDs, and save."""
    label = np.array(Image.open(label_path), dtype=np.int32)
    remapped = np.full_like(label, 255, dtype=np.uint8)
    for orig_id, new_id in REMAP.items():
        remapped[label == orig_id] = new_id
    Image.fromarray(remapped, mode="L").save(output_path)


def find_split_files(rellis_root):
    """Locate train.lst, val.lst, test.lst in the RELLIS-3D directory."""
    root = Path(rellis_root)
    split_files = {}
    for split_name in ["train", "val", "test"]:
        # Try common locations
        candidates = [
            root / f"{split_name}.lst",
            root / "split" / f"{split_name}.lst",
            root / "Rellis-3D" / f"{split_name}.lst",
        ]
        for c in candidates:
            if c.exists():
                split_files[split_name] = c
                break
    return split_files


def parse_split_file(split_file):
    """Parse a RELLIS-3D split .lst file. Returns list of (image_relpath, label_relpath)."""
    pairs = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def discover_pairs(rellis_root):
    """If no split files, discover all image/label pairs and create a default split."""
    root = Path(rellis_root)
    # Look for the Rellis-3D subdirectory
    data_root = root / "Rellis-3D" if (root / "Rellis-3D").exists() else root

    pairs = []
    for seq_dir in sorted(data_root.iterdir()):
        if not seq_dir.is_dir() or not seq_dir.name.isdigit():
            continue
        img_dir = seq_dir / "pylon_camera_node"
        lbl_dir = seq_dir / "pylon_camera_node_label_id"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for img_file in sorted(img_dir.glob("*.jpg")):
            # Label file has same stem but .png
            lbl_file = lbl_dir / (img_file.stem + ".png")
            if lbl_file.exists():
                pairs.append((str(img_file), str(lbl_file)))

    # Default split: 70% train, 15% val, 15% test
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    n_train = int(0.7 * len(pairs))
    n_val = int(0.15 * len(pairs))
    splits = {
        "train": [pairs[i] for i in indices[:n_train]],
        "val": [pairs[i] for i in indices[n_train:n_train + n_val]],
        "test": [pairs[i] for i in indices[n_train + n_val:]],
    }
    return splits


def main():
    parser = argparse.ArgumentParser(description="Preprocess RELLIS-3D for SegFormer training")
    parser.add_argument("--rellis-root", required=True, help="Path to RELLIS-3D root directory")
    parser.add_argument("--output", required=True, help="Output directory for processed dataset")
    args = parser.parse_args()

    rellis_root = Path(args.rellis_root)
    output_root = Path(args.output)

    print(f"RELLIS-3D root: {rellis_root}")
    print(f"Output: {output_root}")
    print(f"Classes: {NUM_CLASSES} (void → 255 ignore index)")
    print(f"Remap table: {REMAP}")
    print()

    # Try to find split files
    split_files = find_split_files(rellis_root)

    if split_files:
        print(f"Found split files: {list(split_files.keys())}")
        data_root = rellis_root / "Rellis-3D" if (rellis_root / "Rellis-3D").exists() else rellis_root
        splits = {}
        for split_name, split_file in split_files.items():
            raw_pairs = parse_split_file(split_file)
            # Resolve relative paths
            resolved = []
            for img_rel, lbl_rel in raw_pairs:
                img_path = data_root / img_rel if not os.path.isabs(img_rel) else Path(img_rel)
                lbl_path = data_root / lbl_rel if not os.path.isabs(lbl_rel) else Path(lbl_rel)
                if img_path.exists() and lbl_path.exists():
                    resolved.append((str(img_path), str(lbl_path)))
                else:
                    # Try without leading slash or with different prefix
                    img_path2 = rellis_root / img_rel.lstrip("/")
                    lbl_path2 = rellis_root / lbl_rel.lstrip("/")
                    if img_path2.exists() and lbl_path2.exists():
                        resolved.append((str(img_path2), str(lbl_path2)))
            splits[split_name] = resolved
    else:
        print("No split files found. Discovering pairs and creating default split...")
        splits = discover_pairs(rellis_root)

    # Process each split
    for split_name, pairs in splits.items():
        print(f"\n--- {split_name}: {len(pairs)} pairs ---")
        if not pairs:
            print(f"  WARNING: No pairs found for {split_name}!")
            continue

        img_out = output_root / split_name / "images"
        lbl_out = output_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in tqdm(pairs, desc=f"  Processing {split_name}"):
            img_path = Path(img_path)
            lbl_path = Path(lbl_path)

            # Use a unique name: sequence_frame
            stem = f"{img_path.parent.parent.name}_{img_path.stem}"

            # Copy image
            dst_img = img_out / f"{stem}.jpg"
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Remap and save label
            dst_lbl = lbl_out / f"{stem}.png"
            if not dst_lbl.exists():
                remap_label(lbl_path, dst_lbl)

    # Write class mappings
    id2label = {str(k): v for k, v in CONTIGUOUS_ID2NAME.items()}
    label2id = {v: k for k, v in CONTIGUOUS_ID2NAME.items()}

    with open(output_root / "id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)
    with open(output_root / "label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)

    print(f"\nDone! Output in {output_root}")
    print(f"Classes ({NUM_CLASSES}):")
    for i, name in CONTIGUOUS_ID2NAME.items():
        print(f"  {i}: {name}")
    print(f"  255: void (ignore)")


if __name__ == "__main__":
    main()
