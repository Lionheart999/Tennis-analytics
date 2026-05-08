"""
Cut dead time from a tennis match video using the trained GRU classifier.

Usage:
    python scripts/cut_video.py <stem>
    python scripts/cut_video.py W2019_Djokovic_v_Federer_part_018

Output:
    runs/cut/<stem>_cut.mp4

Pipeline:
    1. Load pre-extracted feature CSV (or exit with instructions if missing)
    2. Run GRU inference → per-frame rally probabilities
    3. Threshold + smooth → binary keep/cut array
    4. Merge short gaps, drop short segments
    5. Add buffer around each segment
    6. ffmpeg -c copy to extract + concatenate kept segments
"""

import sys
import pickle
import subprocess
import tempfile
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT         = Path(__file__).parent.parent
FEATURES_DIR = ROOT / 'features'
VIDEOS_DIR   = ROOT / 'processed_videos'
DATASET_DIR  = ROOT / 'gru_dataset'
MODEL_DIR    = ROOT / 'models'
OUT_DIR      = ROOT / 'runs' / 'cut'

sys.path.insert(0, str(ROOT / 'scripts'))
from train_gru      import RallyGRU
from build_dataset  import FEATURE_COLS, WINDOW
from predict_video  import load_model, predict

# ── Tunable parameters ─────────────────────────────────────────────────────────
THRESHOLD    = 0.5    # probability above which a frame is rally
MIN_RALLY_S  = 1.0    # drop rally segments shorter than this (seconds)
MERGE_GAP_S  = 1.0    # merge two rally segments if gap between them is ≤ this
BUFFER_S     = 0.5    # seconds to keep before and after each rally segment


def get_segments(probs, fps):
    """
    Convert a per-frame probability array into a list of (start_s, end_s) segments
    after applying threshold, gap merging, minimum length, and buffer.
    """
    keep   = probs >= THRESHOLD
    n      = len(keep)

    # ── Find contiguous keep-runs ──────────────────────────────────────────────
    segments = []
    in_seg   = False
    for i in range(n):
        if keep[i] and not in_seg:
            start = i
            in_seg = True
        elif not keep[i] and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, n - 1))

    # ── Merge gaps ─────────────────────────────────────────────────────────────
    merge_frames = int(MERGE_GAP_S * fps)
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= merge_frames:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))
    segments = merged

    # ── Drop short segments ────────────────────────────────────────────────────
    min_frames = int(MIN_RALLY_S * fps)
    segments   = [s for s in segments if s[1] - s[0] >= min_frames]

    # ── Add buffer ─────────────────────────────────────────────────────────────
    buf = int(BUFFER_S * fps)
    segments = [
        (max(0, s[0] - buf), min(n - 1, s[1] + buf))
        for s in segments
    ]

    # Convert to seconds
    return [(s[0] / fps, s[1] / fps) for s in segments]


def cut_with_ffmpeg(video_path, segments, out_path):
    """Extract segments with ffmpeg -c copy, then concatenate."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        part_files = []

        for i, (start_s, end_s) in enumerate(segments):
            part = tmp / f'part_{i:04d}.mp4'
            subprocess.run(
                ['ffmpeg', '-y',
                 '-ss', f'{start_s:.3f}',
                 '-to', f'{end_s:.3f}',
                 '-i',  str(video_path),
                 '-c:v', 'copy',
                 '-c:a', 'aac',
                 '-reset_timestamps', '1',
                 str(part)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            part_files.append(part)

        concat_txt = tmp / 'concat.txt'
        concat_txt.write_text('\n'.join(f"file '{p}'" for p in part_files))

        subprocess.run(
            ['ffmpeg', '-y',
             '-f',    'concat',
             '-safe', '0',
             '-i',    str(concat_txt),
             '-c',    'copy',
             str(out_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


def main():
    stem     = sys.argv[1] if len(sys.argv) > 1 else 'W2019_Djokovic_v_Federer_part_018'
    video    = VIDEOS_DIR   / f'{stem}.mp4'
    feat_csv = FEATURES_DIR / f'{stem}_features.csv'
    out_path = OUT_DIR / f'{stem}_cut.mp4'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not feat_csv.exists():
        print(f"ERROR: No feature CSV found for {stem}")
        print(f"Run:  python scripts/extract_features_batch.py  first")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Clip:   {stem}")
    print(f"Device: {device}")

    with open(DATASET_DIR / 'scaler.pkl', 'rb') as fh:
        scaler = pickle.load(fh)

    model = load_model(device)

    print("Running inference ...", end=' ', flush=True)
    frame_idx_arr, probs = predict(feat_csv, scaler, model, device)
    print("done")

    # Get video fps from feature CSV timestamp column
    df  = pd.read_csv(feat_csv, usecols=['frame_idx', 'timestamp_s'])
    fps = len(df) / float(df['timestamp_s'].iloc[-1])

    segments = get_segments(probs, fps)

    total_in  = float(df['timestamp_s'].iloc[-1])
    total_out = sum(e - s for s, e in segments)
    print(f"\nOriginal duration:  {total_in/60:.1f} min")
    print(f"Kept duration:      {total_out/60:.1f} min  ({total_out/total_in:.0%})")
    print(f"Rally segments:     {len(segments)}")
    print(f"\nSegments:")
    for i, (s, e) in enumerate(segments):
        sm, ss = divmod(s, 60)
        em, es = divmod(e, 60)
        print(f"  {i+1:3d}.  {int(sm):02d}:{ss:05.2f} → {int(em):02d}:{es:05.2f}  ({e-s:.1f}s)")

    print(f"\nCutting video ...", end=' ', flush=True)
    cut_with_ffmpeg(video, segments, out_path)
    print("done")
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
