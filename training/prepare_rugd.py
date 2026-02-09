#!/usr/bin/env python
"""
Preprocess RUGD for SegFormer fine-tuning.

Converts the raw RUGD download into HuggingFace-compatible format:
  - Converts RGB annotation PNGs to single-channel class ID PNGs
  - Organizes into train/val/test splits with images/ and labels/ subdirs
  - Writes id2label.json and label2id.json

Usage:
    python prepare_rugd.py --rugd-root /path/to/rugd --output /path/to/rugd_processed

Expected input structure (after unzipping):
    rugd/
    ├── RUGD_frames-with-annotations/   (or RUGD_frames/)
    │   ├── creek/
    │   │   ├── creek-00000.png         # RGB frames
    │   │   ├── creek-00005.png
    │   │   └── ...
    │   ├── park-1/
    │   │   └── ...
    │   └── ...
    └── RUGD_annotations/               (or RUGD_annotations/)
        ├── creek/
        │   ├── creek-00000.png         # RGB-colored annotation PNGs
        │   ├── creek-00005.png
        │   └── ...
        ├── park-1/
        │   └── ...
        └── ...
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# RUGD class definitions: name → RGB color (from RUGD annotation colormap)
# 24 classes including void
RUGD_CLASSES = [
    ("void",      (0, 0, 0)),
    ("dirt",      (108, 64, 20)),
    ("sand",      (255, 229, 204)),
    ("grass",     (0, 102, 0)),
    ("tree",      (0, 255, 0)),
    ("bush",      (0, 153, 153)),
    ("concrete",  (0, 128, 255)),
    ("mud",       (0, 0, 255)),
    ("gravel",    (255, 255, 0)),
    ("water",     (255, 0, 127)),
    ("asphalt",   (64, 64, 64)),
    ("mulch",     (255, 128, 0)),
    ("pole",      (255, 0, 0)),
    ("object",    (153, 76, 0)),
    ("building",  (102, 102, 0)),
    ("log",       (102, 0, 0)),
    ("person",    (0, 255, 128)),
    ("fence",     (204, 153, 255)),
    ("rock-bed",  (102, 0, 204)),
    ("sign",      (255, 153, 204)),
    ("bridge",    (0, 102, 102)),
    ("trash",     (153, 204, 255)),
    ("barrier",   (102, 255, 255)),
    ("bicycle",   (255, 255, 127)),
]

# Build lookup: RGB tuple → class ID
# void (ID 0) → 255 (ignore index); rest → contiguous 0..22
RGB_TO_ORIG_ID = {rgb: i for i, (_, rgb) in enumerate(RUGD_CLASSES)}

# Contiguous mapping: skip void, assign 0..22 to the 23 non-void classes
NON_VOID_CLASSES = [(name, rgb) for i, (name, rgb) in enumerate(RUGD_CLASSES) if i != 0]
CONTIGUOUS_ID2NAME = {i: name for i, (name, _) in enumerate(NON_VOID_CLASSES)}
CONTIGUOUS_NAME2ID = {v: k for k, v in CONTIGUOUS_ID2NAME.items()}

# RGB → contiguous ID (void RGB → 255)
RGB_TO_CONTIGUOUS = {}
for orig_id, (name, rgb) in enumerate(RUGD_CLASSES):
    if orig_id == 0:
        RGB_TO_CONTIGUOUS[rgb] = 255  # void → ignore
    else:
        RGB_TO_CONTIGUOUS[rgb] = orig_id - 1  # shift down by 1

NUM_CLASSES = len(CONTIGUOUS_ID2NAME)

# RUGD train/val/test split (from the RUGD paper)
# Format: sequence_name → split
RUGD_SPLITS = {
    "creek":    "test",     # reserved for test (rock bed terrain)
    "park-1":   "train",
    "park-2":   "train",
    "park-8":   "train",
    "trail":    "train",
    "trail-1":  "train",
    "trail-2":  "train",
    "trail-3":  "train",
    "trail-4":  "train",
    "trail-5":  "train",
    "trail-6":  "val",
    "trail-7":  "test",     # reserved for test (blurry frames)
    "trail-8":  "train",
    "trail-9":  "train",
    "trail-10": "train",
    "trail-11": "train",
    "trail-12": "train",
    "village":  "val",
}


def build_rgb_lut(rgb_to_id):
    """Build a fast (256, 256, 256) → uint8 lookup with nearest-neighbor fallback.

    Exact RGB matches are mapped directly. All other RGB values are mapped to
    the nearest known class color (L2 distance), so JPEG artifacts or slight
    color shifts don't produce spurious 255 (ignore) pixels.
    """
    lut = np.full((256, 256, 256), 255, dtype=np.uint8)

    # First pass: exact matches
    for (r, g, b), cid in rgb_to_id.items():
        lut[r, g, b] = cid

    # Second pass: nearest-neighbor for all other entries
    # Build array of known colors and their IDs
    known_colors = np.array(list(rgb_to_id.keys()), dtype=np.float32)  # (N, 3)
    known_ids = np.array(list(rgb_to_id.values()), dtype=np.uint8)     # (N,)

    # For efficiency, only fill in a tolerance band around each known color
    # rather than brute-forcing all 16M entries
    TOLERANCE = 30  # max per-channel distance to consider
    for (r, g, b), cid in rgb_to_id.items():
        r_lo, r_hi = max(0, r - TOLERANCE), min(255, r + TOLERANCE)
        g_lo, g_hi = max(0, g - TOLERANCE), min(255, g + TOLERANCE)
        b_lo, b_hi = max(0, b - TOLERANCE), min(255, b + TOLERANCE)
        for ri in range(r_lo, r_hi + 1):
            for gi in range(g_lo, g_hi + 1):
                for bi in range(b_lo, b_hi + 1):
                    if lut[ri, gi, bi] == 255:
                        # Find nearest known color
                        dists = (known_colors[:, 0] - ri)**2 + \
                                (known_colors[:, 1] - gi)**2 + \
                                (known_colors[:, 2] - bi)**2
                        lut[ri, gi, bi] = known_ids[np.argmin(dists)]
    return lut


_fallback_warned = [False]


def convert_annotation(ann_path, output_path, lut):
    """Convert an RGB annotation PNG to a single-channel class ID PNG."""
    ann = np.array(Image.open(ann_path).convert("RGB"), dtype=np.uint8)
    h, w, _ = ann.shape
    label = lut[ann[:, :, 0], ann[:, :, 1], ann[:, :, 2]]
    # Warn once if any pixels still mapped to 255 (unmapped colors outside tolerance)
    n_unmapped = np.sum(label == 255)
    if n_unmapped > 0 and not _fallback_warned[0]:
        print(f"  WARNING: {n_unmapped} pixels in {ann_path.name} mapped to 255 (unknown color).")
        print(f"  This may indicate annotation colors differ from expected. Check RUGD_CLASSES.")
        _fallback_warned[0] = True
    Image.fromarray(label, mode="L").save(output_path)


def find_data_dirs(rugd_root):
    """Locate the frames and annotations directories."""
    root = Path(rugd_root)

    # Try common directory names
    frames_candidates = [
        root / "RUGD_frames-with-annotations",
        root / "RUGD_frames",
        root,
    ]
    ann_candidates = [
        root / "RUGD_annotations",
        root / "RUGD_annotations-colormap",
        root,
    ]

    frames_dir = None
    for c in frames_candidates:
        if c.exists() and any(c.iterdir()):
            # Check if it has sequence subdirs
            if any((c / seq).is_dir() for seq in RUGD_SPLITS):
                frames_dir = c
                break

    ann_dir = None
    for c in ann_candidates:
        if c.exists() and any(c.iterdir()):
            if any((c / seq).is_dir() for seq in RUGD_SPLITS):
                ann_dir = c
                break

    return frames_dir, ann_dir


def main():
    parser = argparse.ArgumentParser(description="Preprocess RUGD for SegFormer training")
    parser.add_argument("--rugd-root", required=True, help="Path to RUGD root directory")
    parser.add_argument("--output", required=True, help="Output directory for processed dataset")
    args = parser.parse_args()

    rugd_root = Path(args.rugd_root)
    output_root = Path(args.output)

    print(f"RUGD root: {rugd_root}")
    print(f"Output: {output_root}")
    print(f"Classes: {NUM_CLASSES} (void → 255 ignore index)")
    print()

    frames_dir, ann_dir = find_data_dirs(rugd_root)

    if frames_dir is None:
        print("ERROR: Could not find frames directory. Expected subdirectories like 'creek', 'park-1', etc.")
        print(f"  Searched in: {rugd_root}")
        return 1
    if ann_dir is None:
        print("ERROR: Could not find annotations directory.")
        print(f"  Searched in: {rugd_root}")
        return 1

    print(f"Frames dir: {frames_dir}")
    print(f"Annotations dir: {ann_dir}")

    # Build RGB lookup table
    lut = build_rgb_lut(RGB_TO_CONTIGUOUS)

    # Collect pairs per split
    split_pairs = {"train": [], "val": [], "test": []}

    for seq_name, split_name in sorted(RUGD_SPLITS.items()):
        seq_frames = frames_dir / seq_name
        seq_anns = ann_dir / seq_name

        if not seq_frames.exists():
            print(f"  WARNING: Frames dir not found for sequence '{seq_name}', skipping")
            continue
        if not seq_anns.exists():
            print(f"  WARNING: Annotations dir not found for sequence '{seq_name}', skipping")
            continue

        # Match frames to annotations by filename
        ann_files = {f.stem: f for f in seq_anns.glob("*.png")}
        for frame_file in sorted(seq_frames.glob("*.png")):
            if frame_file.stem in ann_files:
                split_pairs[split_name].append((
                    frame_file,
                    ann_files[frame_file.stem],
                    f"{seq_name}_{frame_file.stem}",
                ))

    # Process each split
    for split_name, pairs in split_pairs.items():
        print(f"\n--- {split_name}: {len(pairs)} pairs ---")
        if not pairs:
            print(f"  WARNING: No pairs found for {split_name}!")
            continue

        img_out = output_root / split_name / "images"
        lbl_out = output_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for frame_path, ann_path, stem in tqdm(pairs, desc=f"  Processing {split_name}"):
            # Copy frame
            dst_img = img_out / f"{stem}.png"
            if not dst_img.exists():
                shutil.copy2(frame_path, dst_img)

            # Convert annotation
            dst_lbl = lbl_out / f"{stem}.png"
            if not dst_lbl.exists():
                convert_annotation(ann_path, dst_lbl, lut)

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
    import sys
    sys.exit(main() or 0)
