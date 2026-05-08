"""
Build sliding-window GRU dataset from SimCLR CNN embeddings.

Same windowing / val-split logic as build_dataset.py, but reads 512-d
embeddings instead of 34 hand-engineered features. The GRU architecture
is unchanged — only the input dimensionality differs.

Output saved to gru_dataset_cnn/:
    X_train.npy  — (N, 75, 512) float32
    y_train.npy  — (N,) float32
    X_val.npy
    y_val.npy
    scaler.pkl   — StandardScaler fitted on train embeddings

Usage:
    python scripts/build_dataset_cnn.py
    python scripts/train_gru.py --dataset gru_dataset_cnn --model rally_gru_cnn
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

ROOT           = Path(__file__).parent.parent
FEATURES_DIR   = ROOT / 'features'
EMBEDDINGS_DIR = ROOT / 'embeddings'
LABELS_DIR     = ROOT / 'labels'
VIDEOS_DIR     = ROOT / 'processed_videos'
RAW_VIDEO_DIR  = ROOT / 'raw_videos'
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
    """Hold out the last clip of each match for val (same logic as build_dataset.py)."""
    stems = sorted(
        c.stem.replace('_features', '')
        for c in FEATURES_DIR.glob('*_features.csv')
        if (LABELS_DIR / (c.stem.replace('_features', '') + '_segments.csv')).exists()
        and (EMBEDDINGS_DIR / (c.stem.replace('_features', '') + '_embeddings.npy')).exists()
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


def load_clip(stem):
    seg_csv = LABELS_DIR     / f'{stem}_segments.csv'
    emb_arr = np.load(EMBEDDINGS_DIR / f'{stem}_embeddings.npy')   # (n, 512)
    idx_arr = np.load(EMBEDDINGS_DIR / f'{stem}_frame_idx.npy')    # (n,)

    video   = find_video(stem)
    labels, _ = expand_labels(seg_csv, video, sample_every=1)

    label_vec = np.array([
        int(labels[int(i)]) if int(i) < len(labels) else 0
        for i in idx_arr
    ])

    return emb_arr, label_vec


def make_windows(emb_arr, label_vec):
    n    = len(emb_arr)
    X, y = [], []
    for start in range(0, n - WINDOW + 1, STRIDE):
        center = start + WINDOW // 2
        lbl    = label_vec[center]
        if lbl == -1:
            continue
        X.append(emb_arr[start:start + WINDOW])
        y.append(float(lbl))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_clips, val_clips = split_clips()

    print(f"Train clips ({len(train_clips)}): {train_clips}")
    print(f"Val clips   ({len(val_clips)}):   {val_clips}\n")

    X_parts = {'train': [], 'val': []}
    y_parts = {'train': [], 'val': []}

    for stem in val_clips:
        print(f"  [VAL]   {stem} ...", end=' ', flush=True)
        emb, lbl = load_clip(stem)
        X, y     = make_windows(emb, lbl)
        X_parts['val'].append(X)
        y_parts['val'].append(y)
        print(f"{len(y)} windows  rally={y.sum():.0f}")

    for stem in train_clips:
        emb, lbl = load_clip(stem)
        X, y     = make_windows(emb, lbl)
        is_full_match = '_part_' not in stem
        if is_full_match:
            split_idx = int(len(X) * 0.8)
            X_parts['train'].append(X[:split_idx])
            y_parts['train'].append(y[:split_idx])
            X_parts['val'].append(X[split_idx:])
            y_parts['val'].append(y[split_idx:])
            print(f"  [SPLIT] {stem}: {split_idx} train / {len(X)-split_idx} val")
        else:
            print(f"  [TRAIN] {stem} ...", end=' ', flush=True)
            X_parts['train'].append(X)
            y_parts['train'].append(y)
            print(f"{len(y)} windows  rally={y.sum():.0f}")

    X_train = np.concatenate(X_parts['train'])
    y_train = np.concatenate(y_parts['train'])
    X_val   = np.concatenate(X_parts['val'])
    y_val   = np.concatenate(y_parts['val'])

    n,  t,  f  = X_train.shape
    nv, tv, fv = X_val.shape
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, f)).reshape(n,  t,  f)
    X_val   = scaler.transform    (X_val.reshape  (-1, fv)).reshape(nv, tv, fv)

    np.save(DATASET_DIR / 'X_train.npy', X_train)
    np.save(DATASET_DIR / 'y_train.npy', y_train)
    np.save(DATASET_DIR / 'X_val.npy',   X_val)
    np.save(DATASET_DIR / 'y_val.npy',   y_val)
    with open(DATASET_DIR / 'scaler.pkl', 'wb') as fh:
        pickle.dump(scaler, fh)

    print(f"\nX_train: {X_train.shape}  rally={y_train.mean():.1%}")
    print(f"X_val:   {X_val.shape}    rally={y_val.mean():.1%}")
    print(f"Embedding dim: {f}")
    print(f"Saved to {DATASET_DIR}")


if __name__ == '__main__':
    main()
