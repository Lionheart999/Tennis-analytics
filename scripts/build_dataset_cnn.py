"""
Build the CNN GRU dataset index — no large .npy files written.

Instead of pre-materialising every window, this script saves:
    gru_dataset_cnn/train_index.pkl  — list of (emb_path, start, label)
    gru_dataset_cnn/val_index.pkl
    gru_dataset_cnn/scaler.pkl       — StandardScaler fitted on train frames

train_gru_cnn.py uses EmbeddingWindowDataset (defined here) to read windows
on-the-fly from the embedding memmaps, so STRIDE=5 works without disk pressure.

Usage:
    python scripts/build_dataset_cnn.py
    python scripts/train_gru_cnn.py
"""

import sys
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

ROOT           = Path(__file__).parent.parent
EMBEDDINGS_DIR = ROOT / 'embeddings'
LABELS_DIR     = ROOT / 'labels'
VIDEOS_DIR     = ROOT / 'processed_videos'
RAW_VIDEO_DIR  = ROOT / 'raw_videos' / 'Converted'
DATASET_DIR    = ROOT / 'gru_dataset_cnn'

sys.path.insert(0, str(ROOT / 'scripts'))
from label_segments import expand_labels

WINDOW = 75
STRIDE = 5


def find_video(stem):
    for d in (VIDEOS_DIR, RAW_VIDEO_DIR):
        p = d / f'{stem}.mp4'
        if p.exists():
            return p
    return None


def split_clips():
    stems = sorted(
        s.stem.replace('_segments', '')
        for s in LABELS_DIR.glob('*_segments.csv')
        if (EMBEDDINGS_DIR / (s.stem.replace('_segments', '') + '_embeddings.npy')).exists()
    )
    by_match = defaultdict(list)
    for stem in stems:
        match = stem.rsplit('_part_', 1)[0]
        by_match[match].append(stem)

    train, val = [], []
    for match_clips in sorted(by_match.values()):
        if len(match_clips) > 1:
            val.append(match_clips[-1])
            train.extend(match_clips[:-1])
        else:
            train.extend(match_clips)
    return train, val


def build_index(stems, is_train_full_split=False):
    """
    Returns list of (emb_path_str, start_frame, label) for every valid window.
    Full-match clips (no _part_) are split 80/20 when is_train_full_split=True.
    """
    train_idx, val_idx = [], []

    for stem in stems:
        emb_path = EMBEDDINGS_DIR / f'{stem}_embeddings.npy'
        seg_csv  = LABELS_DIR     / f'{stem}_segments.csv'
        video    = find_video(stem)

        emb    = np.load(emb_path, mmap_mode='r')
        labels, _ = expand_labels(seg_csv, video, sample_every=1)
        n      = len(emb)

        positions = []
        for start in range(0, n - WINDOW + 1, STRIDE):
            center = start + WINDOW // 2
            lbl    = int(labels[center]) if center < len(labels) else 0
            if lbl == -1:
                continue
            positions.append((str(emb_path), start, float(lbl)))

        is_full = '_part_' not in stem
        if is_full and is_train_full_split:
            split = int(len(positions) * 0.8)
            train_idx.extend(positions[:split])
            val_idx.extend(positions[split:])
        else:
            train_idx.extend(positions)

        print(f"  {stem}: {len(positions)} windows")

    return train_idx, val_idx


def fit_scaler(train_stems, sample_stride=50):
    """Fit StandardScaler on sampled frames from training clips (not full windows)."""
    scaler = StandardScaler()
    for stem in train_stems:
        emb = np.load(EMBEDDINGS_DIR / f'{stem}_embeddings.npy', mmap_mode='r')
        scaler.partial_fit(emb[::sample_stride])
    return scaler


class EmbeddingWindowDataset(Dataset):
    """
    Reads 75-frame embedding windows on-the-fly from memmap files.
    No large arrays held in RAM — the OS page cache handles repeated access.
    """
    def __init__(self, index, scaler):
        self.index  = index   # list of (emb_path_str, start, label)
        self.scaler = scaler

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        emb_path, start, label = self.index[idx]
        emb    = np.load(emb_path, mmap_mode='r')
        window = emb[start:start + WINDOW].copy()           # (75, 512)
        window = self.scaler.transform(window).astype(np.float32)
        return torch.tensor(window), torch.tensor(label)


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_clips, val_clips = split_clips()

    print(f"Train clips ({len(train_clips)}): {train_clips}")
    print(f"Val clips   ({len(val_clips)}):   {val_clips}\n")

    print("Building val index:")
    val_idx, _ = build_index(val_clips)

    print("\nBuilding train index:")
    train_idx, val_extra = build_index(train_clips, is_train_full_split=True)
    val_idx.extend(val_extra)

    print(f"\nTrain windows: {len(train_idx):,}  Val windows: {len(val_idx):,}")

    print("\nFitting scaler on sampled train frames...")
    scaler = fit_scaler(train_clips)

    with open(DATASET_DIR / 'train_index.pkl', 'wb') as f:
        pickle.dump(train_idx, f)
    with open(DATASET_DIR / 'val_index.pkl', 'wb') as f:
        pickle.dump(val_idx, f)
    with open(DATASET_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    rally_train = sum(lbl for _, _, lbl in train_idx) / len(train_idx)
    rally_val   = sum(lbl for _, _, lbl in val_idx)   / len(val_idx)
    print(f"rally train={rally_train:.1%}  val={rally_val:.1%}")
    print(f"Saved to {DATASET_DIR}")


if __name__ == '__main__':
    main()
