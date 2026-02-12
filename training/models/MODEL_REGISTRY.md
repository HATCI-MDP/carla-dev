# Fine-Tuned Model Registry

Track all fine-tuned models, their configs, and results here.

## How to Use

### SegFormer (HuggingFace)
```bash
# In CARLA:
python scripts/autopilot_segformer.py --model training/models/<folder_name>

# In Python:
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
model = SegformerForSemanticSegmentation.from_pretrained("training/models/<folder_name>")
processor = SegformerImageProcessor.from_pretrained("training/models/<folder_name>")
```

### DeepLabV3+ (PyTorch .pth)
```python
# Requires: torch, torchvision
import torch
model = torch.load("training/models/deeplabv3_orfd/best_model.pth", map_location="cpu")
# Binary output: road (1) vs not-road (0)
```

---

## Model Comparison

| Folder | Dataset | Size | Base Pretrained | Resolution | Epochs | Batch | LR | mIoU | Mean Acc | Pixel Acc | Platform | Notes |
|--------|---------|------|-----------------|------------|--------|-------|----|------|----------|-----------|----------|-------|
| `rellis3d_b0_ade_arc` | RELLIS-3D | B0 | ADE20K | 256 | 50 | 4 | 6e-5 | 0.484 | 0.595 | 0.844 | ARC (V100) | fp16, batch=1 x grad_accum=4 |
| `rugd_b0_ade_arc` | RUGD | B0 | ADE20K | 256 | 50 | 4 | 6e-5 | 0.341 | 0.438 | 0.858 | ARC (V100) | fp16, batch=1 x grad_accum=4 |
| `rugd_b2_cityscapes_colab` | RUGD | B2 | Cityscapes | 512 | 50 | 8 | 6e-5 | 0.370 | 0.469 | 0.872 | Colab (A100) | some classes have 0 predictions |
| `rellis3d_b2_ade_colab` | RELLIS-3D | B2 | ADE20K | 768 | ~32/50 | 8 | 6e-5 | 0.447 | 0.572 | 0.870 | Colab (A100) | incomplete (checkpoint-26500), ran out of compute |

> **Fill in results as runs complete.** Add new rows for each new training run.

### DeepLabV3+ Models

| Folder | Dataset | Architecture | Resolution | IoU (val) | IoU (test) | Dice | Notes |
|--------|---------|-------------|------------|-----------|------------|------|-------|
| `deeplabv3_orfd` | ORFD + RELLIS | DeepLabV3+ ResNet50 | 512 | 0.858 | 0.930 | 0.963 | Binary (road vs not-road), .pth format |

> **Note:** DeepLabV3+ IoU is binary segmentation — not comparable to multi-class SegFormer mIoU.

---

## Key Findings

- **B0 @ 256** on ARC: constrained by 10GB RAM limit (lsa2 account). Decent baseline but low resolution hurts accuracy.
- **Cityscapes vs ADE20K base**: Cityscapes has 19 urban classes; ADE20K has 150 classes including outdoor/terrain (tree, grass, earth, dirt, water, rock, sand). **ADE20K is likely better for off-road datasets.**
- **Resolution is the biggest lever** for accuracy — 256→512 should give a significant mIoU boost.
- **B2 vs B5**: B5 gives ~2-5% mIoU over B2. Try B2 + ADE20K + 512 before jumping to B5.
- **Learning rate**: B2 may benefit from lower LR (2e-5 vs 6e-5) to avoid overfitting.

## Suggested Next Runs

1. **B2 + ADE20K + 512** on both datasets (highest expected improvement)
2. **B2 + ADE20K + 512 + LR 2e-5** if overfitting observed
3. **B5 + ADE20K + 512** if B2 results plateau

---

## Dataset Info

| Dataset | Classes | Train Samples | Val Samples | Task | Source |
|---------|---------|---------------|-------------|------|--------|
| RELLIS-3D | 21 (+ void=255) | 3,302 | 983 | Multi-class terrain | Texas A&M off-road robot |
| RUGD | 23 (+ void=255) | ~5,000 | ~1,400 | Multi-class terrain | Unstructured ground robot |
| ORFD | 2 (road/not-road) | — | — | Binary road detection | Off-Road Freespace Detection |

## Base Model IDs

| Base | Model ID |
|------|----------|
| ADE20K B0 | `nvidia/segformer-b0-finetuned-ade-512-512` |
| ADE20K B2 | `nvidia/segformer-b2-finetuned-ade-512-512` |
| ADE20K B5 | `nvidia/segformer-b5-finetuned-ade-640-640` |
| Cityscapes B0 | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` |
| Cityscapes B2 | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` |
| Cityscapes B5 | `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` |
