#!/usr/bin/env python
"""
Fine-tune SegFormer for semantic segmentation on RELLIS-3D or RUGD.

Expects preprocessed dataset (from prepare_rellis3d.py or prepare_rugd.py):
    dataset_dir/
    ├── train/images/  train/labels/
    ├── val/images/    val/labels/
    ├── id2label.json
    └── label2id.json

Usage:
    python train_segformer.py \
        --dataset-dir /path/to/rellis3d_processed \
        --model-name nvidia/segformer-b2-finetuned-ade-512-512 \
        --output-dir /path/to/outputs/rellis3d_segformer_b2 \
        --epochs 50 --batch-size 4 --lr 6e-5

The trained model is saved in HuggingFace format and can be loaded directly
in the CARLA scripts with: --model /path/to/outputs/rellis3d_segformer_b2
"""

import argparse
import json
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer,
)


class SegmentationDataset(Dataset):
    """Dataset that loads image/label pairs from directories."""

    def __init__(self, image_dir, label_dir, processor, max_size=512):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.processor = processor
        self.max_size = max_size

        # Match images to labels by stem
        image_files = {f.stem: f for f in self.image_dir.iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png")}
        label_files = {f.stem: f for f in self.label_dir.iterdir()
                       if f.suffix.lower() == ".png"}

        common_stems = sorted(set(image_files.keys()) & set(label_files.keys()))
        self.pairs = [(image_files[s], label_files[s]) for s in common_stems]

        if not self.pairs:
            raise ValueError(
                f"No matching image/label pairs found in:\n"
                f"  images: {image_dir}\n"
                f"  labels: {label_dir}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path)  # single-channel, uint8 class IDs

        # Resize both to max_size (SegFormer expects consistent input)
        image = image.resize((self.max_size, self.max_size), Image.BILINEAR)
        label = label.resize((self.max_size, self.max_size), Image.NEAREST)

        # Process image with SegFormer processor
        encoded = self.processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)  # (3, H, W)

        # Label as tensor
        label_array = np.array(label, dtype=np.int64)
        labels = torch.from_numpy(label_array)  # (H, W)

        return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics_factory(num_classes, ignore_index=255):
    """Return a compute_metrics function for the HuggingFace Trainer."""
    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # logits: (N, num_classes, H', W') — may be smaller than label size
        # labels: (N, H, W)

        logits_tensor = torch.from_numpy(logits)
        # Upsample logits to label size
        labels_h, labels_w = labels.shape[1], labels.shape[2]
        logits_upsampled = torch.nn.functional.interpolate(
            logits_tensor, size=(labels_h, labels_w),
            mode="bilinear", align_corners=False,
        )
        preds = logits_upsampled.argmax(dim=1).numpy()  # (N, H, W)

        # Flatten and filter ignore index
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)
        mask = labels_flat != ignore_index
        preds_flat = preds_flat[mask]
        labels_flat = labels_flat[mask]

        results = metric.compute(
            predictions=preds_flat,
            references=labels_flat,
            num_labels=num_classes,
            ignore_index=ignore_index,
        )
        return {
            "mean_iou": results["mean_iou"],
            "mean_accuracy": results["mean_accuracy"],
            "overall_accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SegFormer on a preprocessed off-road dataset"
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Path to preprocessed dataset (with train/, val/, id2label.json)",
    )
    parser.add_argument(
        "--model-name", default="nvidia/segformer-b2-finetuned-ade-512-512",
        help="HuggingFace model ID to fine-tune from (default: SegFormer-B2 ADE20K)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save trained model and checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate (default: 6e-5)")
    parser.add_argument("--image-size", type=int, default=512, help="Input image size (default: 512)")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (fp16)")
    parser.add_argument("--dataloader-workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Load class mappings
    id2label_path = dataset_dir / "id2label.json"
    label2id_path = dataset_dir / "label2id.json"
    if not id2label_path.exists() or not label2id_path.exists():
        print(f"ERROR: id2label.json / label2id.json not found in {dataset_dir}")
        print("Run the preprocessing script first (prepare_rellis3d.py or prepare_rugd.py).")
        return 1

    with open(id2label_path) as f:
        id2label = json.load(f)
    with open(label2id_path) as f:
        label2id = json.load(f)

    num_classes = len(id2label)
    print(f"Dataset: {dataset_dir}")
    print(f"Classes: {num_classes}")
    print(f"id2label: {id2label}")
    print(f"Base model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load processor and model
    print("Loading processor and model...")
    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    # Don't reduce labels — we handle our own mapping
    processor.do_reduce_labels = False

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # classifier head size will change
    )

    # Create datasets
    print("Loading datasets...")
    train_dataset = SegmentationDataset(
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "labels",
        processor,
        max_size=args.image_size,
    )
    val_dataset = SegmentationDataset(
        dataset_dir / "val" / "images",
        dataset_dir / "val" / "labels",
        processor,
        max_size=args.image_size,
    )
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="polynomial",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=args.dataloader_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_factory(num_classes),
    )

    # Train
    print("\n=== Starting training ===\n")
    trainer.train()

    # Save final model
    print(f"\nSaving final model to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Also save id2label/label2id alongside the model for easy reference
    import shutil
    shutil.copy2(id2label_path, Path(args.output_dir) / "id2label.json")
    shutil.copy2(label2id_path, Path(args.output_dir) / "label2id.json")

    # Final evaluation
    print("\n=== Final evaluation on validation set ===\n")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"\nDone! Model saved to: {args.output_dir}")
    print("To use in CARLA:")
    print(f"  python scripts/autopilot_segformer.py --model {args.output_dir}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
