#!/usr/bin/env python3
"""
Phased YOLOv8 training script for tennis analytics (players + ball).

Workflow:
  1) Phase 1 (frozen backbone): adapt neck/head to your domain
  2) [optional] Use Phase 1 weights to generate better labels in CVAT
  3) Phase 2 (unfrozen): fine-tune full network on improved dataset

Usage examples:
  # Phase 1 (frozen backbone)
  python train_yolo_phased.py \
      --phase phase1 \
      --data path/to/data.yaml \
      --base yolov8n.pt \
      --imgsz 640 --batch 32 --epochs 30 --freeze 10 \
      --project runs --name phase1_freeze

  # Phase 2 (unfreeze, start from Phase 1 best)
  python train_yolo_phased.py \
      --phase phase2 \
      --data path/to/data.yaml \
      --weights runs/detect/phase1_freeze/weights/best.pt \
      --imgsz 640 --batch 24 --epochs 80 \
      --lr0 0.003 --optimizer sgd \
      --project runs --name phase2_unfreeze

Notes:
- Assumes `pip install ultralytics` and CUDA are available.
- Script will print a concise summary of settings and save Ultralytics outputs under `--project/--name`.
- Uses Ultralytics Python API (same as CLI under the hood).
"""

import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(
        description="Phased YOLOv8 training (phase1 freeze → annotate more → phase2 unfreeze)"
    )

    phase = p.add_argument_group("Phase & paths")
    phase.add_argument("--phase", choices=["phase1", "phase2"], required=True,
                       help="phase1: freeze backbone, phase2: unfreeze full network")
    phase.add_argument("--data", required=True, help="Path to data.yaml")
    phase.add_argument("--base", default="yolov8n.pt",
                       help="Base model to start from in phase1 (ignored for phase2 if --weights is set)")
    phase.add_argument("--weights", default=None,
                       help="Starting weights for phase2 (e.g., runs/detect/phase1_freeze/weights/best.pt)")

    train = p.add_argument_group("Training hyperparams")
    train.add_argument("--imgsz", type=int, default=640, help="Train image size")
    train.add_argument("--batch", type=int, default=32, help="Batch size")
    train.add_argument("--epochs", type=int, default=50, help="Epochs")
    train.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    train.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"], help="Optimizer")
    train.add_argument("--freeze", type=int, default=10, help="#layers to freeze in phase1 (ignored in phase2)")
    train.add_argument("--device", default="0", help="CUDA device id, e.g. '0' or '0,1' or 'cpu'")
    train.add_argument("--patience", type=int, default=20, help="Early stop patience (epochs)")
    train.add_argument("--augment", action="store_true", help="Enable stronger default augmentation")
    train.add_argument("--cos_lr", action="store_true", help="Use cosine LR scheduler")

    io = p.add_argument_group("IO & logging")
    io.add_argument("--project", default="runs", help="Ultralytics project directory")
    io.add_argument("--name", default=None, help="Run name (subfolder under project)")
    io.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    io.add_argument("--seed", type=int, default=67, help="Random seed for reproducibility")
    io.add_argument("--exist_ok", action="store_true", help="Allow existing project/name (no increment)")

    return p.parse_args()


def train_phase1(args):
    print("[Phase 1] Frozen-backbone fine-tuning")
    print(f"  base:    {args.base}")
    print(f"  data:    {args.data}")
    print(f"  imgsz:   {args.imgsz}, batch: {args.batch}, epochs: {args.epochs}")
    print(f"  freeze:  {args.freeze}, lr0: {args.lr0}, opt: {args.optimizer}")

    model = YOLO(args.base)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        freeze=args.freeze,
        lr0=args.lr0,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name or "phase1_freeze",
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        cos_lr=args.cos_lr,
        # Stronger augs toggle; Ultralytics has rich defaults already
        augment=args.augment,
        exist_ok=args.exist_ok,
    )
    return results


def train_phase2(args):
    print("[Phase 2] Unfrozen full-model fine-tuning")
    if not args.weights:
        print("ERROR: --weights required for phase2 (use best.pt from phase1)")
        sys.exit(2)

    print(f"  weights: {args.weights}")
    print(f"  data:    {args.data}")
    print(f"  imgsz:   {args.imgsz}, batch: {args.batch}, epochs: {args.epochs}")
    print(f"  lr0:     {args.lr0}, opt: {args.optimizer}")

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name or "phase2_unfreeze",
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        cos_lr=args.cos_lr,
        augment=args.augment,
        exist_ok=args.exist_ok,
    )
    return results


def main():
    args = parse_args()

    # Basic checks
    if not Path(args.data).exists():
        print(f"ERROR: data.yaml not found: {args.data}")
        sys.exit(2)

    os.environ.setdefault("ULTRALYTICS_VERBOSE", "True")

    if args.phase == "phase1":
        _ = train_phase1(args)
    else:
        _ = train_phase2(args)

    print("\nDone. Check your runs under:")
    run_dir = Path(args.project) / "detect" / (args.name or ("phase1_freeze" if args.phase == "phase1" else "phase2_unfreeze"))
    print(f"  {run_dir.resolve()}")


if __name__ == "__main__":
    main()
