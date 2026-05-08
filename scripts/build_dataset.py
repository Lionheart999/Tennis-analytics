"""
Build sliding-window dataset from feature CSVs and segment label CSVs.

Output (saved to dataset/):
    X_train.npy  — (N, 75, n_features) float32
    y_train.npy  — (N,) float32  0=non-rally  1=rally
    X_val.npy
    y_val.npy
    scaler.pkl   — fitted StandardScaler (apply to inference data)

Val split: last clip of each match held out.
Excluded frames (label=-1) are dropped from both splits.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

ROOT         = Path(__file__).parent.parent
FEATURES_DIR = ROOT / 'features'
LABELS_DIR   = ROOT / 'labels'
VIDEOS_DIR   = ROOT / 'processed_videos'
DATASET_DIR  = ROOT / 'gru_dataset'

sys.path.insert(0, str(ROOT / 'scripts'))
from label_segments import expand_labels

WINDOW = 75   # frames (~3s at 25fps)
STRIDE = 5    # step between windows

FEATURE_COLS = [
    # Ball
    'ball_detected', 'ball_conf', 'ball_speed',
    'ball_freq_2s', 'ball_freq_5s', 'ball_streak_norm', 'time_since_ball_s',
    # Players — magnitudes
    'p1_speed', 'p2_speed', 'combined_speed',
    'p1_accel', 'p2_accel',
    # Players — directional
    'p1_vx', 'p1_vy', 'p2_vx', 'p2_vy',
    'p1_ax', 'p1_ay', 'p2_ax', 'p2_ay',
    # Players — rolling windows
    'p1_move_intensity', 'p2_move_intensity',
    'p1_avg_speed_3s', 'p2_avg_speed_3s',
    'p1_max_speed_3s', 'p2_max_speed_3s',
    'p1_std_speed_3s', 'p2_std_speed_3s',
    'time_since_move_s',
    # Court / frame
    'both_players_visible', 'zoom_level', 'separation',
    'frame_diff', 'camera_cut',
]


def split_clips():
    """Hold out the last (highest part number) clip of each match for val."""
    stems = sorted(
        c.stem.replace('_features', '')
        for c in FEATURES_DIR.glob('*_features.csv')
        if (LABELS_DIR / (c.stem.replace('_features', '') + '_segments.csv')).exists()
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
    feat_csv = FEATURES_DIR / f'{stem}_features.csv'
    seg_csv  = LABELS_DIR   / f'{stem}_segments.csv'
    video    = VIDEOS_DIR   / f'{stem}.mp4'
    if not video.exists():
        video = ROOT / 'raw_videos' / f'{stem}.mp4'

    df = pd.read_csv(feat_csv).fillna(0.0)

    # Derive acceleration components from velocity if not already in CSV
    if 'p1_ax' not in df.columns:
        df['p1_ax'] = df['p1_vx'].diff().fillna(0.0)
        df['p1_ay'] = df['p1_vy'].diff().fillna(0.0)
        df['p2_ax'] = df['p2_vx'].diff().fillna(0.0)
        df['p2_ay'] = df['p2_vy'].diff().fillna(0.0)

    labels, _ = expand_labels(seg_csv, video, sample_every=1)

    df['label'] = df['frame_idx'].apply(
        lambda i: int(labels[int(i)]) if int(i) < len(labels) else 0
    )
    return df


def make_windows(df):
    feat = df[FEATURE_COLS].values.astype(np.float32)
    lbls = df['label'].values
    n    = len(feat)
    X, y = [], []
    for start in range(0, n - WINDOW + 1, STRIDE):
        center = start + WINDOW // 2
        lbl    = lbls[center]
        if lbl == -1:
            continue
        X.append(feat[start:start + WINDOW])
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
        df   = load_clip(stem)
        X, y = make_windows(df)
        X_parts['val'].append(X)
        y_parts['val'].append(y)
        print(f"{len(y)} windows  rally={y.sum():.0f}  non-rally={(1-y).sum():.0f}")

    for stem in train_clips:
        df   = load_clip(stem)
        X, y = make_windows(df)
        is_full_match = '_part_' not in stem
        if is_full_match:
            split_idx = int(len(X) * 0.8)
            X_parts['train'].append(X[:split_idx])
            y_parts['train'].append(y[:split_idx])
            X_parts['val'].append(X[split_idx:])
            y_parts['val'].append(y[split_idx:])
            print(f"  [SPLIT] {stem}: {split_idx} train / {len(X)-split_idx} val windows  rally={y.mean():.1%}")
        else:
            print(f"  [TRAIN] {stem} ...", end=' ', flush=True)
            X_parts['train'].append(X)
            y_parts['train'].append(y)
            print(f"{len(y)} windows  rally={y.sum():.0f}  non-rally={(1-y).sum():.0f}")

    X_train = np.concatenate(X_parts['train'])
    y_train = np.concatenate(y_parts['train'])
    X_val   = np.concatenate(X_parts['val'])
    y_val   = np.concatenate(y_parts['val'])

    # Normalise: fit on train only, apply to both
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
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Saved to {DATASET_DIR}")


if __name__ == '__main__':
    main()
