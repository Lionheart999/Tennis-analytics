"""
SimCLR self-supervised pre-training on raw tennis match videos.

Trains a ResNet-18 encoder with NT-Xent loss. Two independently augmented
views of the same frame form a positive pair; all other frames in the batch
are negatives. Uses full raw match videos (not just labeled clips) for
maximum pre-training data.

Usage:
    python scripts/simclr_pretrain.py

Saves:
    models/simclr_encoder.pt  — encoder-only state dict (no projection head)

Trains on all non-validation matches. Where both a regular and h264 version
exist, the h264 version is used. Validation matches are excluded.
"""

import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
import cv2
from pathlib import Path

ROOT          = Path(__file__).parent.parent
RAW_VIDEO_DIR = ROOT / 'raw_videos'
MODEL_DIR     = ROOT / 'models'

VAL_MATCHES = {
    'A2025_Sinner_v_Shelton',
    'R2025_Musetti_v_Tiafoe',
    'W2019_Federer_v_Nadal',
}

SEED         = 42
BATCH        = 256      # NT-Xent needs large batches
EPOCHS       = 50
LR           = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE  = 0.5
FRAME_STRIDE = 5        # sample every 5th frame (~5 fps from 25 fps source)
IMG_SIZE     = 224
PATIENCE     = 8


def base_match(stem):
    """Strip _h264 and _part_NNN suffixes to get the canonical match name."""
    stem = re.sub(r'_h264$', '', stem)
    stem = re.sub(r'_part_\d+$', '', stem)
    return stem


def select_videos():
    """
    One video per non-validation match. Where both regular and h264 exist,
    prefer the h264 version (more widely compatible codec).
    """
    by_match = {}
    for vp in sorted(RAW_VIDEO_DIR.glob('*.mp4')):
        b = base_match(vp.stem)
        if b in VAL_MATCHES:
            continue
        # h264 version overwrites regular version (sort order ensures this)
        if b not in by_match or vp.stem.endswith('_h264'):
            by_match[b] = vp
    return sorted(by_match.values())


def get_aug():
    blur_k = 11   # ~10% of 112px
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=blur_k, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class SimCLRVideoDataset(IterableDataset):
    """
    Reads each video sequentially (no seeking). Each worker handles a
    disjoint subset of videos, so total I/O is distributed across workers.
    Returns (view1, view2) pairs of augmented frames.
    """
    def __init__(self, video_paths, frame_stride=FRAME_STRIDE):
        self.video_paths  = list(video_paths)
        self.frame_stride = frame_stride
        self.aug = get_aug()

    def _iter_video(self, vp):
        cap = cv2.VideoCapture(str(vp))
        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_count % self.frame_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield self.aug(rgb), self.aug(rgb)
            frame_count += 1
        cap.release()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        paths = self.video_paths

        if worker_info is not None:
            # Each worker processes its own disjoint subset of videos
            paths = [p for i, p in enumerate(paths)
                     if i % worker_info.num_workers == worker_info.id]

        for vp in paths:
            yield from self._iter_video(vp)


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.encoder   = nn.Sequential(*list(backbone.children())[:-1])   # → (B, 512, 1, 1)
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)     # (B, 512)
        z = self.projector(h)
        return F.normalize(z, dim=1)        # (B, 128) on unit hypersphere


def nt_xent(z1, z2, temp):
    """NT-Xent loss: z1[i] and z2[i] are positive pairs; all cross-pairs are negatives."""
    N  = z1.shape[0]
    z  = torch.cat([z1, z2])                # (2N, 128)
    s  = torch.matmul(z, z.T) / temp        # (2N, 2N) cosine sims
    s.fill_diagonal_(float('-inf'))          # exclude self-similarity
    labels = torch.cat([torch.arange(N, 2*N),
                        torch.arange(N)]).to(z.device)
    return F.cross_entropy(s, labels)


def estimate_frames(video_paths, frame_stride):
    total = 0
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // frame_stride
        cap.release()
    return total


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    video_paths = select_videos()
    print(f"Training videos ({len(video_paths)}):")
    for vp in video_paths:
        print(f"  {vp.name}")

    print(f"\nEstimating dataset size...", end=' ', flush=True)
    n_frames = estimate_frames(video_paths, FRAME_STRIDE)
    print(f"{n_frames:,} frames  (stride={FRAME_STRIDE})")
    n_workers = min(len(video_paths), 4)
    print(f"Workers: {n_workers}  (one video subset per worker)\n")

    dataset = SimCLRVideoDataset(video_paths)
    loader  = DataLoader(
        dataset, batch_size=BATCH,
        num_workers=n_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    model     = SimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss      = float('inf')
    patience_count = 0

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Best':>8}")
    print("-" * 28)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, n_batches = 0.0, 0

        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            loss = nt_xent(model(v1), model(v2), TEMPERATURE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total     += loss.item()
            n_batches += 1

        if n_batches == 0:
            print(f"Epoch {epoch}: no batches — check video paths.")
            break

        avg_loss = total / n_batches
        scheduler.step()

        marker = " ✓" if avg_loss < best_loss else ""
        print(f"{epoch:>5}  {avg_loss:>8.4f}  {best_loss:>8.4f}{marker}")

        if avg_loss < best_loss:
            best_loss      = avg_loss
            patience_count = 0
            torch.save(model.encoder.state_dict(), MODEL_DIR / 'simclr_encoder.pt')
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Encoder saved to {MODEL_DIR / 'simclr_encoder.pt'}")


if __name__ == '__main__':
    main()
