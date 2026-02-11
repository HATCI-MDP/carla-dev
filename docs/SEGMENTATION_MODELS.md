# Segmentation models: ADE20K and OffSeg

This guide covers how to run **ADE20K** (via SegFormer) and **OffSeg** (off-road segmentation) in this project.

---

## ADE20K (SegFormer)

**What it is:** SegFormer trained on ADE20K: 150 classes (sky, road, tree, grass, earth, building, etc.). Good for general scene parsing and off-road–style classes (earth, path, grass, road). Full class list (id → name): [ADE20K_CLASSES.txt](ADE20K_CLASSES.txt).

**Do I need to download the model?** No. Install with `pip install -r requirements-segmentation.txt` only. The first time you run a script that uses the model, Hugging Face will download it automatically (requires internet). No manual download step.

### 1. Install (once)

From the project root with your venv activated:

```powershell
pip install -r requirements-segmentation.txt
```

You also need `opencv-python` for the CARLA script (likely already installed). If not: `pip install opencv-python`.

### 2. Run on a single image

```powershell
python scripts/run_segformer_image.py path/to/image.jpg
```

Output is saved as `segmentation_output.png` (or use `--output out.png`). This confirms the model and dependencies work.

### 3. Run in CARLA (live ego camera)

With CARLA running (CarlaUE4.exe and a map loaded), run:

```powershell
python scripts/autopilot_segformer.py
```

This spawns a vehicle on autopilot, attaches a camera, and runs SegFormer on each frame (script: `autopilot_segformer.py`). You get a window with **camera | segmentation** side-by-side. Press **q** or **ESC** to exit.

Options: `--host`, `--port`, `--map Town02`, `--model nvidia/segformer-b0-finetuned-ade-512-512` (default), `--width 640`, `--height 480`. Smaller resolution = faster inference.

### 4. Accuracy vs speed (SegFormer)

**Ways to get higher accuracy:**

| Change | Effect | Cost |
|--------|--------|------|
| **Larger model** | B0 → B2 → B5: more capacity, better mIoU on ADE20K | Slower inference, more GPU memory |
| **Higher resolution** | e.g. 640×480 or 512×512 camera; model sees more detail | Slower (more pixels per frame) |

**Model variants (same API, swap `--model`):**

- `nvidia/segformer-b0-finetuned-ade-512-512` — default; fastest, good for real time.
- `nvidia/segformer-b2-finetuned-ade-512-512` — better accuracy, slower.
- `nvidia/segformer-b5-finetuned-ade-640-640` — highest accuracy (e.g. on ADE20K), slowest; needs a strong GPU.

**Inference rate (FPS)** is the opposite of accuracy in practice: higher FPS usually means smaller model or lower resolution. To improve accuracy, use a larger model and/or higher `--width` / `--height` and accept lower FPS (or use `--infer-every` to keep the camera smooth).

**Lower-spec machines (MacBook, no GPU):** Use `--width 320 --height 240 --infer-every 5` to reduce load. Inference runs in a background thread by default so the camera stays smooth.

```powershell
python scripts/autopilot_segformer.py --width 320 --height 240 --infer-every 5
```

**Manual drive + segmentation:** Use `manual_control_segformer.py`. Works with both ADE20K pretrained and fine-tuned off-road models. Requires `pynput` (included in `requirements-segmentation.txt`).

```powershell
python scripts/manual_control_segformer.py
python scripts/manual_control_segformer.py --model training/models/rellis3d_segformer_b0
```

Controls: **W** = throttle, **S** = brake, **A** = left, **D** = right, **SPACE** = hand brake, **Q** = toggle reverse. Press **q** or **ESC** in the window to exit.

**FPS and speed:** All SegFormer CARLA scripts (autopilot and manual) show **FPS** and **Speed (km/h)** in the top-left of the window. No option to disable.

**Does SegFormer detect only some classes?** No. The model always predicts **all 150 ADE20K classes** in one forward pass (one label per pixel). You can’t turn off classes to save compute. You *can* post-process: only *display* or *use* certain class IDs (e.g. mask to road + grass + path). Our legend already filters to driving-relevant classes for display only; the full 150 are still predicted.

### 4. Other ADE20K variants (image script)

```powershell
# B2 – better accuracy, slower
python scripts/run_segformer_image.py path/to/image.jpg --model nvidia/segformer-b2-finetuned-ade-512-512
```

**Reference:** [SEGFORMER_SETUP.md](SEGFORMER_SETUP.md) has more SegFormer details and code snippets.

---

## OffSeg

**What it is:** Off-road semantic segmentation (paper: [OFFSEG](https://arxiv.org/abs/2103.12417)). Two-stage: (1) 4-class segmentation (sky, traversable, non-traversable, obstacle), (2) sub-classes on traversable region (grass, puddle, dirt, etc.). Trained on RUGD / RELLIS-3D. Best for **off-road / dirt / trails**.

### What you have to do to get OffSeg working

OffSeg is **not** in this repo. You run it from the official OFFSEG repo in a **separate Python env** (they use Python 3.6–3.8 and TensorFlow + PyTorch; keep it away from your CARLA/Python 3.12 venv).

| Step | What to do |
|------|------------|
| 1 | Clone [github.com/kasiv008/OFFSEG](https://github.com/kasiv008/OFFSEG). Go into `OFFSEG/Pipeline`. |
| 2 | Create a dedicated venv (e.g. Python 3.8): `py -3.8 -m venv .venv_offseg`, activate it, then `pip install torch torchvision tensorflow opencv-python numpy pandas Pillow scikit-learn fast-pytorch-kmeans`. |
| 3 | Download pretrained weights from [Google Drive (OFFSEG weights)](https://drive.google.com/drive/folders/1a-DDJ0C6Q4Vfl5100pGQIbMCekD4xpV2?usp=sharing). You need the segmentation checkpoint (e.g. `model_final.pth`) and the classification model (e.g. `keras_model.h5`). |
| 4 | In `OFFSEG/Pipeline/pipeline.py`, set at the top: `img_path` (folder with input images), `final_path` (folder for outputs), `dataset` (path to `model_final.pth`). In the line that loads the Keras model, set the path to your `keras_model.h5`. |
| 5 | From `OFFSEG/Pipeline` run: `python pipeline.py --model bisenetv2 --weight-path C:\path\to\model_final.pth`. It processes all images in `img_path` and writes to `final_path`. For a single image, put that image alone in `img_path`. |

If you hit version errors (e.g. TensorFlow/PyTorch), try matching their [requirements.txt](https://github.com/kasiv008/OFFSEG/blob/main/requirements.txt) versions in that venv.

**How to run it (detailed):**

OffSeg is a separate repo with its own environment (Python 3.6–3.8, TensorFlow + PyTorch). Use a **separate virtualenv** so it doesn’t conflict with CARLA (Python 3.12).

1. **Clone the repo and go to Pipeline:**

   ```powershell
   cd C:\path\to\your\repos
   git clone https://github.com/kasiv008/OFFSEG.git
   cd OFFSEG\Pipeline
   ```

2. **Create a dedicated venv** (e.g. Python 3.8):

   ```powershell
   py -3.8 -m venv .venv_offseg
   .venv_offseg\Scripts\activate
   pip install torch torchvision tensorflow opencv-python numpy pandas Pillow scikit-learn fast-pytorch-kmeans
   ```

   (Their [requirements.txt](https://github.com/kasiv008/OFFSEG/blob/main/requirements.txt) pins old versions; you can try current PyTorch/TF or match theirs if you hit errors.)

3. **Download pre-trained weights:**

   - RUGD weights (recommended): [Google Drive – OFFSEG pretrained weights](https://drive.google.com/drive/folders/1a-DDJ0C6Q4Vfl5100pGQIbMCekD4xpV2?usp=sharing)
   - Save the segmentation model (e.g. `model_final.pth`) and the classification model (e.g. `keras_model.h5`) to a folder you’ll reference below.

4. **Configure paths in `Pipeline/pipeline.py`:**

   Edit the top of `pipeline.py` and set:

   - `img_path` – directory containing input images (e.g. a folder with one or more `.jpg`)
   - `final_path` – directory where output segmentation images are saved
   - `dataset` – path to `model_final.pth`
   - Classification model: in the line `model = tf.keras.models.load_model(...)` set the path to your `keras_model.h5` (often in `OFFSEG/classification/model/` if you have that folder from their repo/setup).

5. **Run the pipeline:**

   From `OFFSEG/Pipeline`:

   ```powershell
   python pipeline.py --model bisenetv2 --weight-path C:\path\to\model_final.pth
   ```

   It will process every image in `img_path` and write results to `final_path`.

**Single image:** Put one image in a folder and set that folder as `img_path`; run the same command.

**Use in CARLA:** Run OffSeg in a separate process or port their inference (PyTorch net + optional TF classifier) into a CARLA camera callback; input = RGB frame, output = 4-class (or sub-class) mask for traversability.

**References:**

- Repo: [github.com/kasiv008/OFFSEG](https://github.com/kasiv008/OFFSEG)
- Weights: [Google Drive](https://drive.google.com/drive/folders/1a-DDJ0C6Q4Vfl5100pGQIbMCekD4xpV2?usp=sharing)

### Easier alternatives to OffSeg (same style as ADE20K)

If OffSeg’s setup (separate repo, venv, TensorFlow + PyTorch, manual paths) is too much, you can stay in this project and use models that install with a single `pip install` and run like ADE20K:

- **SegFormer on Cityscapes** – Same API as ADE20K: `--model nvidia/segformer-b0-finetuned-cityscapes-1024-1024`. Fewer classes, tuned for driving (road, car, person, etc.). Good for on-road; less “dirt/grass” than ADE20K.
- **SegFormer ADE20K** – What you already use. Good for general outdoor + off-road–style (earth, path, grass, road, dirt track). No extra setup.
- **Other Hugging Face segmentation models** – Search [huggingface.co](https://huggingface.co/models?pipeline_tag=image-segmentation) for “semantic segmentation”; many use the same `from_pretrained` pattern. None are “RUGD/OffSeg” out of the box, but ADE20K + Cityscapes cover most driving and outdoor needs without cloning repos or managing a second env.

---

## Summary

| Model   | How to run in this project                    | Best for                          |
|--------|-----------------------------------------------|-----------------------------------|
| **ADE20K** | `pip install -r requirements-segmentation.txt` then `python scripts/run_segformer_image.py image.jpg` or SegFormer CARLA scripts | General scene + off-road–style (earth, path, grass, road) |
| **OffSeg**  | Clone OFFSEG, separate venv, download weights, set paths in `Pipeline/pipeline.py`, run `pipeline.py` on an image folder | Off-road / dirt / trails (traversable vs non-traversable + sub-classes) |

For CARLA integration, start with ADE20K (SegFormer) in the same Python process as the CARLA client; add OffSeg later if you need dedicated off-road classes and can run it in a separate env or adapt their code.
