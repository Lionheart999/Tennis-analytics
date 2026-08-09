"""
Post-processing sensitivity sweep: vary BUFFER_S across [0.25, 0.5, 1.0] seconds.

MIN_RALLY_S and MERGE_GAP_S are held fixed at their defaults (1.0 s each).
Inference runs once per model per match; only post-processing is re-applied.

Usage:
    python scripts/sweep_buffer.py

Output:
    runs/eval/buffer_sweep.csv   — full results
    Console table ready to copy into the thesis
"""

import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

ROOT            = Path(__file__).parent.parent
FEATURES_DIR    = ROOT / 'features'
EMBEDDINGS_DIR  = ROOT / 'embeddings'
LABELS_DIR      = ROOT / 'labels' / 'val'
MODEL_DIR       = ROOT / 'models'
DATASET_DIR_GRU = ROOT / 'gru_dataset'
DATASET_DIR_CNN = ROOT / 'gru_dataset_cnn'
OUT_DIR         = ROOT / 'runs' / 'eval'

sys.path.insert(0, str(ROOT / 'scripts'))
from predict_video import load_model as load_gru_model, predict as predict_gru
from eval_segments_held_out import (
    load_encoder, predict_cnn, load_gt_segments, compare_segments,
)
from build_dataset_cnn import WINDOW as WINDOW_CNN
from train_gru import RallyGRU

# ── Constants ──────────────────────────────────────────────────────────────────
VAL_MATCHES = [
    ('A2025_Sinner_v_Shelton_preview', 'A2025_Sinner_v_Shelton_h264', 'Hard'),
    ('R2025_Musetti_v_Tiafoe_h264',    'R2025_Musetti_v_Tiafoe_h264', 'Clay'),
    ('W2019_Federer_v_Nadal_h264',     'W2019_Federer_v_Nadal_h264',  'Grass'),
]

BUFFER_VALUES = [0.25, 0.5, 1.0]
THRESHOLD     = 0.5
MIN_RALLY_S   = 1.0
MERGE_GAP_S   = 1.0
SMOOTH_WINDOW = 5


# ── Parameterised post-processing ──────────────────────────────────────────────
def get_segments(probs, fps, buffer_s,
                 threshold=THRESHOLD, min_rally_s=MIN_RALLY_S, merge_gap_s=MERGE_GAP_S):
    keep = probs >= threshold
    n    = len(keep)

    segments = []
    in_seg   = False
    for i in range(n):
        if keep[i] and not in_seg:
            start  = i
            in_seg = True
        elif not keep[i] and in_seg:
            segments.append([start, i - 1])
            in_seg = False
    if in_seg:
        segments.append([start, n - 1])

    merge_frames = int(merge_gap_s * fps)
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= merge_frames:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    min_frames = int(min_rally_s * fps)
    merged     = [s for s in merged if s[1] - s[0] >= min_frames]

    buf = int(buffer_s * fps)
    merged = [(max(0, s[0] - buf), min(n - 1, s[1] + buf)) for s in merged]

    return [(s / fps, e / fps) for s, e in merged]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load models
    with open(DATASET_DIR_GRU / 'scaler.pkl', 'rb') as f:
        gru_scaler = pickle.load(f)
    gru_model = load_gru_model(device)

    with open(DATASET_DIR_CNN / 'scaler.pkl', 'rb') as f:
        cnn_scaler = pickle.load(f)
    cnn_model = RallyGRU(input_size=512).to(device)
    cnn_model.load_state_dict(torch.load(
        MODEL_DIR / 'rally_gru_cnn_best.pt', map_location=device, weights_only=True))

    rows = []

    for label_stem, video_stem, surface in VAL_MATCHES:
        seg_csv   = LABELS_DIR   / f'{label_stem}_segments.csv'
        feat_path = FEATURES_DIR / f'{video_stem}_features.csv'
        emb_path  = EMBEDDINGS_DIR / f'{video_stem}_embeddings.npy'
        match_short = label_stem.replace('_preview', '').replace('_h264', '')

        print(f"{'='*60}")
        print(f"{match_short}  [{surface}]")

        if not seg_csv.exists():
            print(f"  SKIP — no label CSV")
            continue

        gt_segs = load_gt_segments(seg_csv)
        gt_min  = sum(e - s for s, e in gt_segs) / 60

        # ── Hand-engineered GRU ────────────────────────────────────────────────
        if feat_path.exists():
            print(f"  Running hand-engineered GRU inference ...", end=' ', flush=True)
            _, probs_gru = predict_gru(feat_path, gru_scaler, gru_model, device)
            df_ts  = pd.read_csv(feat_path, usecols=['timestamp_s'])
            fps_gru = len(df_ts) / float(df_ts['timestamp_s'].iloc[-1])
            print(f"done  (fps={fps_gru:.1f})")

            for buf in BUFFER_VALUES:
                segs = get_segments(probs_gru, fps_gru, buf)
                pred_min = sum(e - s for s, e in segs) / 60
                p, r, f1, fc, pc, fm = compare_segments(gt_segs, segs)
                rows.append(dict(
                    model='Hand-engineered GRU', match=match_short, surface=surface,
                    buffer_s=buf, gt_segments=len(gt_segs), gt_min=round(gt_min, 1),
                    pred_segments=len(segs), pred_min=round(pred_min, 1),
                    extra_min=round(pred_min - gt_min, 1),
                    precision=round(p, 3), recall=round(r, 3), f1=round(f1, 3),
                    fully_captured=fc, partially_cut=pc, fully_missed=fm,
                ))
                print(f"    buffer={buf:.2f}s  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  "
                      f"extra={pred_min - gt_min:+.1f}min  fc={fc}/{len(gt_segs)}")
        else:
            print(f"  Hand-engineered GRU: SKIP — no feature CSV")

        # ── CNN GRU ────────────────────────────────────────────────────────────
        if emb_path.exists():
            print(f"  Running SimCLR GRU inference ...", end=' ', flush=True)
            import cv2
            cap = cv2.VideoCapture(str(ROOT / 'raw_videos' / f'{video_stem}.mp4'))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            sample_every = max(1, round(video_fps / 25.0))
            eff_fps = video_fps / sample_every

            probs_cnn = predict_cnn(emb_path, cnn_scaler, cnn_model, device, sample_every)
            print(f"done  (eff_fps={eff_fps:.1f})")

            for buf in BUFFER_VALUES:
                segs = get_segments(probs_cnn, eff_fps, buf)
                pred_min = sum(e - s for s, e in segs) / 60
                p, r, f1, fc, pc, fm = compare_segments(gt_segs, segs)
                rows.append(dict(
                    model='SimCLR GRU', match=match_short, surface=surface,
                    buffer_s=buf, gt_segments=len(gt_segs), gt_min=round(gt_min, 1),
                    pred_segments=len(segs), pred_min=round(pred_min, 1),
                    extra_min=round(pred_min - gt_min, 1),
                    precision=round(p, 3), recall=round(r, 3), f1=round(f1, 3),
                    fully_captured=fc, partially_cut=pc, fully_missed=fm,
                ))
                print(f"    buffer={buf:.2f}s  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  "
                      f"extra={pred_min - gt_min:+.1f}min  fc={fc}/{len(gt_segs)}")
        else:
            print(f"  SimCLR GRU: SKIP — no embeddings")

    if not rows:
        print("\nNo results.")
        return

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'buffer_sweep.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'Model':<22} {'Surface':<6} {'Buffer':>8}  "
          f"{'P':>6} {'R':>6} {'F1':>6}  {'Extra(min)':>10}")
    print("-" * 72)
    for _, row in df.iterrows():
        print(f"{row['model']:<22} {row['surface']:<6} {row['buffer_s']:>7.2f}s  "
              f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}  "
              f"{row['extra_min']:>+10.1f}")

    # ── Thesis-ready condensed table ───────────────────────────────────────────
    print(f"\n--- Thesis table (F1 by surface and buffer) ---")
    pivot = df.pivot_table(
        index=['model', 'buffer_s'], columns='surface', values='f1'
    ).round(3)
    pivot['Mean'] = pivot.mean(axis=1).round(3)
    print(pivot.to_string())


if __name__ == '__main__':
    main()
