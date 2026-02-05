# SegFormer setup (ADE20K / Cityscapes)

Minimal install and run for SegFormer semantic segmentation (e.g. for CARLA camera feed). **For a short “how do I run ADE20K and OffSeg?” guide, see [SEGMENTATION_MODELS.md](SEGMENTATION_MODELS.md).**

## 1. Install

From the project root with your venv activated:

```powershell
pip install -r requirements-segmentation.txt
```

That installs PyTorch, Hugging Face `transformers`, and `timm` (used by SegFormer). If you need a CUDA build of PyTorch for GPU, install it first from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```powershell
pip install transformers timm
```

## 2. Quick test (no CARLA)

Run SegFormer on a single image to confirm the setup:

```powershell
python scripts/run_segformer_image.py path/to/any/image.jpg
```

That script loads a pretrained SegFormer (e.g. ADE20K or Cityscapes), runs inference, and saves a colorized segmentation image. If that works, the same model can be wired to CARLA frames.

## 3. Model variants (Hugging Face)

Pretrained SegFormer checkpoints (pick one for `model_id`):

| Model ID | Dataset | Resolution | Use case |
|----------|---------|------------|----------|
| `nvidia/mit-b0` | ImageNet (encoder only) | - | Fine-tune yourself |
| `nvidia/segformer-b0-finetuned-ade-512-512` | ADE20K | 512×512 | Many classes (sky, road, tree, etc.) |
| `nvidia/segformer-b2-finetuned-ade-512-512` | ADE20K | 512×512 | Better accuracy, slower |
| `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | Cityscapes | 1024×1024 | Driving (road, car, person, etc.) |

For off-road–style classes (road, vegetation, terrain), **ADE20K** models are a good start (e.g. `nvidia/segformer-b0-finetuned-ade-512-512`). For driving-specific labels, use **Cityscapes**.

## 4. Minimal Python usage

```python
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_id)
model = SegformerForSemanticSegmentation.from_pretrained(model_id)
model.eval()

# Single image (PIL or path)
from PIL import Image
image = Image.open("photo.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# outputs.logits: (1, num_labels, H, W)
logits = outputs.logits  # resize to image size for overlay
```

From here you can map `logits.argmax(1)` to class IDs, then to a color palette and overlay on the CARLA camera frame (see main guide “Hooking segmentation models up to CARLA”).

## 5. Next step: CARLA + SegFormer

A separate script (e.g. `run_autopilot_camera_segmentation.py`) would: connect to CARLA, spawn vehicle + camera as in `autopilot.py`, load SegFormer once, then in the camera callback: convert frame to PIL/RGB → processor → model → colorize mask → overlay and show in OpenCV. That’s the pipeline described in `CARLA_AND_PERCEPTION_GUIDE.md`.
