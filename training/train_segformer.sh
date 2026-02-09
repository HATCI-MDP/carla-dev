#!/bin/bash
#
# SLURM batch script for fine-tuning SegFormer on Great Lakes.
# Configured for the LSA public GPU account (lsa2).
#
# LSA2 limits: 1 GPU, 2 cores, 10 GB RAM, 24 hr max, 1 job at a time.
# We use SegFormer-B0 by default (fits in 10 GB). B2 may work with
# batch-size 1 + fp16 but is tight. B5 will NOT fit.
#
# BEFORE FIRST USE: Edit the WORK path further down to match your scratch dir.
#
# Submit with:
#   sbatch train_segformer.sh [rellis3d|rugd] [b0|b2]
#
# Examples:
#   sbatch train_segformer.sh rellis3d       # Train on RELLIS-3D with B0 (default)
#   sbatch train_segformer.sh rugd            # Train on RUGD with B0
#   sbatch train_segformer.sh rellis3d b2     # Train on RELLIS-3D with B2 (tight on RAM)
#

# ---- SLURM directives (parsed BEFORE bash runs — no variables allowed) ----
#SBATCH --job-name=segformer-train
#SBATCH --account=lsa2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# Uncomment and set your email to get notifications:
# #SBATCH --mail-user=howden@umich.edu

# ============================================================
# EDIT THESE for your setup
# ============================================================
WORK="/scratch/lsa_root/lsa2/$USER/segformer-training"  # lsa2 scratch dir
CONDA_ENV="$WORK/envs/segformer"
SCRIPTS="$WORK/training"
DATASETS="$WORK/datasets"
OUTPUTS="$WORK/outputs"
# ============================================================

# ---- Parse arguments ----
DATASET="${1:-rellis3d}"    # default: rellis3d
MODEL_SIZE="${2:-b0}"       # default: b0 (fits in lsa2's 10 GB RAM limit)

# ---- Validate dataset ----
case "$DATASET" in
    rellis3d)
        DATASET_DIR="$DATASETS/rellis3d_processed"
        ;;
    rugd)
        DATASET_DIR="$DATASETS/rugd_processed"
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'. Use 'rellis3d' or 'rugd'."
        exit 1
        ;;
esac

# ---- Validate model size ----
case "$MODEL_SIZE" in
    b0)
        MODEL_NAME="nvidia/segformer-b0-finetuned-ade-512-512"
        ;;
    b2)
        MODEL_NAME="nvidia/segformer-b2-finetuned-ade-512-512"
        ;;
    b5)
        echo "WARNING: B5 likely won't fit in lsa2's 10 GB RAM limit. Try b0 or b2."
        MODEL_NAME="nvidia/segformer-b5-finetuned-ade-640-640"
        ;;
    *)
        echo "ERROR: Unknown model size '$MODEL_SIZE'. Use 'b0', 'b2', or 'b5'."
        exit 1
        ;;
esac

OUTPUT_DIR="$OUTPUTS/${DATASET}_segformer_${MODEL_SIZE}"

# ---- Print config ----
echo "============================================"
echo "SegFormer Training Job"
echo "============================================"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Dataset:    $DATASET → $DATASET_DIR"
echo "Model:      $MODEL_NAME"
echo "Output:     $OUTPUT_DIR"
echo "Job ID:     $SLURM_JOB_ID"
echo "============================================"

# ---- Load modules ----
module purge
module load python3.10-anaconda/2023.03  # adjust if different on your cluster
module load cuda/11.8.0                   # adjust to match your PyTorch install
module load cudnn/11.8-v8.6.0             # optional but recommended

echo "Loaded modules:"
module list

# ---- Activate conda env ----
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "Python: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"

# ---- Check dataset exists ----
if [ ! -d "$DATASET_DIR/train" ]; then
    echo "ERROR: Dataset not found at $DATASET_DIR"
    echo "Run the preprocessing script first. See docs/GREAT_LAKES_TRAINING.md"
    exit 1
fi

# ---- Create output dir ----
mkdir -p "$OUTPUT_DIR"

# ---- Run training ----
echo ""
echo "Starting training..."
echo ""

python "$SCRIPTS/train_segformer.py" \
    --dataset-dir "$DATASET_DIR" \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 50 \
    --batch-size 2 \
    --lr 6e-5 \
    --image-size 512 \
    --save-steps 500 \
    --eval-steps 500 \
    --fp16 \
    --dataloader-workers 2

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "Date: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================"

exit $EXIT_CODE
