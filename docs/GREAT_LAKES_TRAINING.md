# Fine-Tuning SegFormer on Great Lakes (RELLIS-3D & RUGD)

This guide covers how to fine-tune SegFormer models on off-road datasets (RELLIS-3D and RUGD) using UMich's Great Lakes HPC cluster, then bring the trained models back to your local CARLA setup.

**Why?** The pretrained SegFormer (ADE20K) we use in CARLA has no off-road–specific classes. Fine-tuning on RELLIS-3D and RUGD gives us models that understand dirt roads, mud, gravel, bushes, puddles, etc. — exactly what our HATCI off-road perception stack needs.

---

## Overview

| Dataset | Classes | Images | Size | Best for |
|---------|---------|--------|------|----------|
| **RELLIS-3D** | 20 (incl. void) | 6,235 | ~11 GB images + ~94 MB labels | Mud, puddles, rubble, barriers — Texas A&M off-road trails |
| **RUGD** | 24 | 7,453 | ~5.3 GB frames + annotations | Dirt, gravel, mulch, sand, rock bed — robot trail navigation |

We train two separate SegFormer models (one per dataset) so we can compare which works better for our CARLA dirt road scenarios.

**Recommended model:** SegFormer-B2 (`nvidia/segformer-b2-finetuned-ade-512-512` as initialization). Good accuracy/speed tradeoff — runs ~15-25 FPS in CARLA on a decent GPU. B0 is faster but less accurate; B5 is overkill for real-time.

---

## RELLIS-3D Classes (20)

The raw RELLIS-3D labels use **non-contiguous** IDs. The `prepare_rellis3d.py` script remaps them to contiguous 0–18 (void → 255 ignore index):

| Original ID | Class | → Remapped ID |
|-------------|-------|---------------|
| 0 | void | 255 (ignore) |
| 1 | dirt | 0 |
| 3 | grass | 1 |
| 4 | tree | 2 |
| 5 | pole | 3 |
| 6 | water | 4 |
| 7 | sky | 5 |
| 8 | vehicle | 6 |
| 9 | object | 7 |
| 10 | asphalt | 8 |
| 12 | building | 9 |
| 15 | log | 10 |
| 17 | person | 11 |
| 18 | fence | 12 |
| 19 | bush | 13 |
| 23 | concrete | 14 |
| 27 | barrier | 15 |
| 29 | puddle | 16 |
| 30 | mud | 17 |
| 31 | rubble | 18 |
| 33 | sign | 19 |
| 34 | rock | 20 |

> **19 non-void classes** after remapping (IDs 0–20). Void pixels (original 0) become 255 and are ignored during training.

## RUGD Classes (24)

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | void | 12 | pole |
| 1 | dirt | 13 | object |
| 2 | sand | 14 | building |
| 3 | grass | 15 | log |
| 4 | tree | 16 | person |
| 5 | bush | 17 | fence |
| 6 | concrete | 18 | rock-bed |
| 7 | mud | 19 | sign |
| 8 | gravel | 20 | bridge |
| 9 | water | 21 | trash |
| 10 | asphalt | 22 | barrier |
| 11 | mulch | 23 | bicycle |

> RUGD annotations are RGB PNGs where each color maps to a class. The `prepare_rugd.py` script converts these to single-channel ID PNGs.

---

# Part 1: Initial Setup (One-Time)

Do these steps once when you first get access to Great Lakes.

## 1.1 Great Lakes Access & LSA Accounts

We use the **LSA public accounts** on Great Lakes (free, no application needed for LSA members).

There are three LSA accounts. **For GPU training we use `lsa2`:**

| Account | Partition | GPU? | Limits |
|---------|-----------|------|--------|
| `lsa1` | `standard` | No | 24 cores, 120 GB RAM, no GPU |
| **`lsa2`** | **`gpu`** | **1 GPU** | **2 cores, 10 GB RAM, 24 hr max, 1 job at a time** |
| `lsa3` | `largemem` | No | 36 cores, 180 GB RAM, 168 hr max (high-memory CPU only) |

> **Important:** LSA public accounts are shared college-wide. Jobs may wait in queue. If you need faster turnaround, consider a [Rackham Graduate Student Research Grant](https://rackham.umich.edu/funding/funding-types/rackham-graduate-student-research-grant/) for a paid account.

To verify your access, SSH in and run:
```bash
my_accounts
```
You should see `lsa1`, `lsa2`, `lsa3` listed. If not, email `arc-support@umich.edu` to be added.

### lsa2 constraints for SegFormer training

With only **10 GB RAM** and **2 cores**, we need to be conservative:
- **SegFormer-B0** fits comfortably (batch size 2, fp16)
- **SegFormer-B2** is tight — use batch size 1 + fp16 if you want to try it
- **SegFormer-B5** will NOT fit in 10 GB
- 50 epochs on RELLIS-3D (~6K images) with B0 should finish well within the 24-hour limit

## 1.2 SSH into Great Lakes

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu
```

You'll land in `/home/<uniqname>` (80 GB quota). Your scratch space is at `/scratch/lsa_root/lsa2/<uniqname>`.

## 1.3 Set Up Your Working Directory

Use `/scratch` for datasets and training (fast I/O, larger quota). `/home` is for scripts/configs only.

```bash
# Create project directory in your lsa2 scratch space
mkdir -p /scratch/lsa_root/lsa2/$USER/segformer-training
cd /scratch/lsa_root/lsa2/$USER/segformer-training
```

> **Tip:** Add a shortcut to your `~/.bashrc`:
> ```bash
> export WORK=/scratch/lsa_root/lsa2/$USER/segformer-training
> ```
> Then `source ~/.bashrc` and use `$WORK` everywhere.

## 1.4 Upload Training Scripts

From your **local machine**, upload the training scripts from this repo:

```bash
# From your local carla-dev directory
scp -r training/ <uniqname>@greatlakes.arc-ts.umich.edu:/scratch/lsa_root/lsa2/<uniqname>/segformer-training/
# This puts files at $WORK/training/ on the cluster
```

Or clone the repo on Great Lakes:

```bash
cd $WORK
git clone <your-repo-url> carla-dev
# If you clone, scripts are at carla-dev/training/ — adjust SCRIPTS path in train_segformer.sh
```

## 1.5 Load Modules & Create Conda Environment

Great Lakes doesn't have `conda` available by default — you must load the Anaconda module first:

```bash
# IMPORTANT: conda won't work until you load the Anaconda module.
# The default "python" module conflicts with it, so purge first:
module purge
module load python3.10-anaconda/2023.03
module load cuda/11.8.0
```

Now create a dedicated conda env in your scratch (not `/home` — saves home quota):

```bash
conda create --prefix $WORK/envs/segformer python=3.10 -y

# conda activate won't work until you initialize the shell hook:
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/segformer

# Install PyTorch with CUDA (match the CUDA module version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install training dependencies
pip install -r $WORK/training/requirements-training.txt
```

> **Module names vary between clusters.** If `python3.10-anaconda/2023.03` doesn't exist, use whatever shows up in `module avail python` that contains "anaconda". Common options: `python3.10-anaconda/2023.03`, `python3.11-anaconda/2024.02`. For CUDA: `cuda/11.8.0`, `cuda/12.1.1`.

## 1.6 Download Datasets

### RELLIS-3D

Download from Google Drive (the image + ID annotation files):

```bash
cd $WORK
mkdir -p datasets/rellis3d
cd datasets/rellis3d

# Option A: Use gdown (install once: pip install gdown)
pip install gdown

# Full Images (~11 GB)
gdown 1F3Leu0H_m6aPVpZITragfreO_SGtL2yV -O Rellis-3D-image.zip

# Full Image Annotations ID Format (~94 MB)
gdown 16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5 -O Rellis-3D-annotation-id.zip

# Image Split File
gdown 1zHmnVaItcYJAWat3Yti1W_5Nfux194WQ -O Rellis-3D-split.zip

# Unzip
unzip Rellis-3D-image.zip
unzip Rellis-3D-annotation-id.zip
unzip Rellis-3D-split.zip
```

> **If gdown fails** (Google Drive rate limits): download on your local machine and `scp` or use [Globus](https://documentation.its.umich.edu/node/5019) to transfer to Great Lakes.

### RUGD

```bash
cd $WORK
mkdir -p datasets/rugd
cd datasets/rugd

# Raw Frames with Annotations (~5.3 GB)
wget http://rugd.vision/data/RUGD_frames-with-annotations.zip -O RUGD_frames-with-annotations.zip

# RGB Annotation Files (~58.7 MB)
wget http://rugd.vision/data/RUGD_annotations.zip -O RUGD_annotations.zip

# Unzip (the RUGD zips may trigger a false "zip bomb" warning — override it)
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip RUGD_frames-with-annotations.zip
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip RUGD_annotations.zip
```

## 1.7 Preprocess Datasets

Both datasets need preprocessing to produce HuggingFace-compatible `image/` and `label/` directories with contiguous class IDs.

```bash
cd $WORK

# Activate env if not already
conda activate $WORK/envs/segformer

# Preprocess RELLIS-3D
python training/prepare_rellis3d.py \
    --rellis-root datasets/rellis3d \
    --output datasets/rellis3d_processed

# Preprocess RUGD
python training/prepare_rugd.py \
    --rugd-root datasets/rugd \
    --output datasets/rugd_processed
```

Each produces:
```
<dataset>_processed/
├── train/
│   ├── images/    # RGB PNGs
│   └── labels/    # Single-channel PNGs (class IDs, void=255)
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── id2label.json  # {0: "dirt", 1: "grass", ...}
└── label2id.json  # {"dirt": 0, "grass": 1, ...}
```

---

# Part 2: Training & Recurring Use

After initial setup, this is your workflow each time you want to train or retrain.

## 2.1 Submit a Training Job

**Do NOT train in the OnDemand terminal or on login nodes.** Login nodes are shared and will kill long-running processes. Use **SLURM batch jobs** (`sbatch`).

```bash
cd $WORK

# FIRST TIME ONLY: verify the WORK path in the script matches your scratch dir
nano training/train_segformer.sh

# Submit the job (already configured for lsa2)
sbatch training/train_segformer.sh
```

The script is pre-configured for `lsa2` (1 GPU, 2 cores, 10 GB RAM, 24 hr). It loads modules, activates your conda env, and runs `train_segformer.py`.

### What the SLURM script does

```
1. Requests 1 GPU node (lsa2 / gpu partition), 2 CPUs, 10 GB RAM, 24-hour wall time
2. Loads CUDA + Python modules
3. Activates your conda env
4. Runs train_segformer.py with your chosen dataset and hyperparameters
5. Saves checkpoints + final model to $WORK/outputs/
```

### Training both models

lsa2 only allows **1 job at a time**, so train them sequentially:

```bash
# Train on RELLIS-3D (submit, wait for it to finish)
sbatch training/train_segformer.sh rellis3d

# After the first job completes, train on RUGD
sbatch training/train_segformer.sh rugd
```

## 2.2 Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch output in real time (job ID from sbatch output)
tail -f slurm-<JOBID>.out

# Check GPU utilization (if you have an interactive session)
srun --jobid=<JOBID> --pty nvidia-smi
```

### Common job states

| State | Meaning |
|-------|---------|
| `PD` | Pending — waiting for resources |
| `R` | Running |
| `CG` | Completing — finishing up |
| `F` | Failed — check the `.out` file for errors |

## 2.3 Retrieve Trained Models

After training completes, the model is saved in HuggingFace format:

```
$WORK/outputs/<dataset>_segformer_b0/
├── config.json
├── model.safetensors  (or pytorch_model.bin)
├── preprocessor_config.json
├── id2label.json
└── training_args.bin
```

**Download to your local machine:**

```bash
# From your LOCAL machine
scp -r <uniqname>@greatlakes.arc-ts.umich.edu:/scratch/lsa_root/lsa2/<uniqname>/segformer-training/outputs/rellis3d_segformer_b0 ./models/rellis3d_segformer_b0

scp -r <uniqname>@greatlakes.arc-ts.umich.edu:/scratch/lsa_root/lsa2/<uniqname>/segformer-training/outputs/rugd_segformer_b0 ./models/rugd_segformer_b0
```

## 2.4 Use in CARLA

Once downloaded, point any of our CARLA segformer scripts at the local model directory:

```powershell
# On your local Windows machine with CARLA
python scripts/autopilot_segformer.py --model ./models/rellis3d_segformer_b0
python scripts/autopilot_segformer.py --model ./models/rugd_segformer_b0
```

The `--model` flag accepts both HuggingFace model IDs and local directory paths. The model's `id2label.json` will be used for the legend automatically.

> **Note:** The class palette and legend in the CARLA scripts will auto-adapt to whatever `id2label` the model config provides. No code changes needed.

## 2.5 Retraining / Hyperparameter Tuning

To retrain with different settings, edit variables in `train_segformer.sh` or pass them as arguments:

```bash
# Different model size (b0 = default, b2 = more accurate but tight on lsa2)
sbatch training/train_segformer.sh rellis3d b2

# The Python script also accepts direct overrides:
# (but prefer editing train_segformer.sh for batch jobs)
python training/train_segformer.py \
    --dataset-dir datasets/rellis3d_processed \
    --model-name nvidia/segformer-b0-finetuned-ade-512-512 \
    --epochs 50 \
    --batch-size 2 \
    --lr 6e-5 \
    --output-dir outputs/rellis3d_segformer_b0
```

### Suggested hyperparameters (for lsa2: 1 GPU, 10 GB RAM)

| Parameter | RELLIS-3D | RUGD | Notes |
|-----------|-----------|------|-------|
| **Base model** | `nvidia/segformer-b0-finetuned-ade-512-512` | same | B0 fits in 10 GB; B2 is tight |
| **Epochs** | 50–100 | 50–100 | RELLIS is smaller, may need more |
| **Batch size** | 2 | 2 | lsa2 has only 10 GB RAM; use 1 for B2 |
| **Learning rate** | 6e-5 | 6e-5 | Standard for fine-tuning SegFormer |
| **Image size** | 512×512 | 512×512 | Matches pretrained resolution |
| **Wall time** | ~6–12 hrs | ~8–16 hrs | B0, 50 epochs, batch 2 (24 hr limit) |

## 2.6 Scratch Cleanup

Great Lakes scratch is **not backed up** and old files may be purged. After training:

1. Download your trained models to your local machine (section 2.3)
2. Optionally keep datasets on scratch for future retraining
3. Delete old checkpoints you don't need: `rm -rf $WORK/outputs/*/checkpoint-*`

---

# Troubleshooting

| Issue | Solution |
|-------|----------|
| `sbatch: error: invalid account` | Verify with `my_accounts`. For LSA, use `lsa2` for GPU jobs. |
| Job stuck in `PD` (pending) | lsa2 GPU partition is shared college-wide. Check: `squeue -A lsa2 -t running`. Wait — jobs may queue for hours. |
| `CUDA out of memory` | Reduce `--batch-size` to 1, or use B0 instead of B2. lsa2 has only 10 GB RAM. |
| `gdown` fails / Google Drive limit | Download datasets locally, transfer via `scp` or Globus |
| `ModuleNotFoundError` | Make sure conda env is activated in your SLURM script. Check paths. |
| Model loads but CARLA legend is wrong | The model's `config.json` has `id2label`. Verify it matches the dataset's class names. |
| OnDemand terminal killed my job | **Don't train in OnDemand terminal.** Use `sbatch` — it runs on compute nodes and survives disconnects. |
| `module: command not found` | You're not on a Great Lakes node. SSH in first. |

---

# File Reference

All training files are in `training/` in this repo:

| File | Purpose |
|------|---------|
| `training/train_segformer.py` | Main fine-tuning script (HuggingFace Trainer) |
| `training/train_segformer.sh` | SLURM batch script — submit with `sbatch` |
| `training/prepare_rellis3d.py` | Preprocess RELLIS-3D → HuggingFace format |
| `training/prepare_rugd.py` | Preprocess RUGD → HuggingFace format |
| `training/requirements-training.txt` | Python dependencies for training |
| `docs/GREAT_LAKES_TRAINING.md` | This guide |
