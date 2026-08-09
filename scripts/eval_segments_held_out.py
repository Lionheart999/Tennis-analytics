"""
Segment-level evaluation of both GRU models on the three held-out matches.

Produces segment-level precision/recall/F1 and fully_captured/partially_cut/
fully_missed counts, matching the format of runs/results.csv.

Usage:
    python scripts/eval_segments_held_out.py

Appends results to runs/results.csv under model names 'gru_held_out' and
'cnn_gru_held_out'.
"""

import sys
import pickle
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
from pathlib import Path

ROOT          = Path(__file__).parent.parent
LABELS_DIR    = ROOT / 'labels' / 'val'
RAW_VIDEO_DIR = ROOT / 'raw_videos'
EMBEDDINGS_DIR= ROOT / 'embeddings'
FEATURES_DIR  = ROOT / 'features'
MODEL_DIR     = ROOT / 'models'
DATASET_DIR_CNN = ROOT / 'gru_dataset_cnn'
DATASET_DIR_GRU = ROOT / 'gru_dataset'
RESULTS_CSV   = ROOT / 'runs' / 'results.csv'

sys.path.insert(0, str(ROOT / 'scripts'))
from train_gru      import RallyGRU
from build_dataset  import FEATURE_COLS, WINDOW
from build_dataset_cnn import WINDOW as WINDOW_CNN
from cut_video      import get_segments, THRESHOLD, MIN_RALLY_S, MERGE_GAP_S, BUFFER_S
from predict_video  import predict as predict_gru, load_model as load_gru_model

VAL_MATCHES = [
    # (label_stem,                       video_stem,                    surface)
    ('A2025_Sinner_v_Shelton_preview', 'A2025_Sinner_v_Shelton_h264', 'Hard'),
    ('R2025_Musetti_v_Tiafoe_h264',    'R2025_Musetti_v_Tiafoe_h264', 'Clay'),
    ('W2019_Federer_v_Nadal_h264',     'W2019_Federer_v_Nadal_h264',  'Grass'),
]

SMOOTH_WINDOW = 5
IMG_SIZE      = 224


# ── CNN embedding helpers ────────────────────────────────────────────────────

def get_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_encoder(device):
    backbone = models.resnet18(weights=None)
    encoder  = nn.Sequential(*list(backbone.children())[:-1])
    sd = torch.load(MODEL_DIR / 'simclr_encoder.pt', map_location=device, weights_only=True)
    encoder.load_state_dict(sd)
    encoder.eval()
    return encoder.to(device)


def predict_cnn(emb_path, scaler, model, device, sample_every=1):
    """Per-frame rally probabilities from CNN embeddings."""
    emb_full   = np.load(emb_path, mmap_mode='r')
    emb        = emb_full[::sample_every].copy()
    emb_scaled = scaler.transform(emb).astype(np.float32)
    n          = len(emb_scaled)

    half = WINDOW_CNN // 2
    pad  = np.concatenate([emb_scaled[:half][::-1], emb_scaled, emb_scaled[-half:][::-1]])

    probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, 512):
            end   = min(start + 512, n)
            batch = np.stack([pad[i:i + WINDOW_CNN] for i in range(start, end)])
            x     = torch.tensor(batch).to(device)
            probs.append(torch.sigmoid(model(x)).cpu().numpy())

    probs  = np.concatenate(probs)
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    return np.convolve(probs, kernel, mode='same')


# ── Segment comparison ───────────────────────────────────────────────────────

def load_gt_segments(seg_csv):
    """Returns list of (start_s, end_s) for rally segments only."""
    df = pd.read_csv(seg_csv)
    rally = df[df['label'] == 'rally'] if 'label' in df.columns else df
    return list(zip(rally['start_s'], rally['end_s']))


def interval_union_duration(intervals):
    """Total duration of a list of (start, end) intervals (handles overlaps)."""
    if not intervals:
        return 0.0
    segs = sorted(intervals)
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged)


def overlap_duration(a_segs, b_segs):
    """Total overlapping time between two lists of segments."""
    total = 0.0
    for a0, a1 in a_segs:
        for b0, b1 in b_segs:
            total += max(0.0, min(a1, b1) - max(a0, b0))
    return total


def compare_segments(gt_segs, pred_segs):
    gt_total   = sum(e - s for s, e in gt_segs)
    pred_total = sum(e - s for s, e in pred_segs)
    tp         = overlap_duration(gt_segs, pred_segs)

    precision = tp / pred_total if pred_total > 0 else 0.0
    recall    = tp / gt_total   if gt_total   > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    fully_captured = partially_cut = fully_missed = 0
    for g0, g1 in gt_segs:
        g_dur     = g1 - g0
        g_overlap = sum(max(0.0, min(g1, p1) - max(g0, p0)) for p0, p1 in pred_segs)
        coverage  = g_overlap / g_dur if g_dur > 0 else 0.0
        if coverage >= 0.9:
            fully_captured += 1
        elif coverage > 0:
            partially_cut += 1
        else:
            fully_missed += 1

    return precision, recall, f1, fully_captured, partially_cut, fully_missed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    with open(DATASET_DIR_CNN / 'scaler.pkl', 'rb') as f:
        cnn_scaler = pickle.load(f)
    cnn_model = RallyGRU(input_size=512).to(device)
    cnn_model.load_state_dict(torch.load(
        MODEL_DIR / 'rally_gru_cnn_best.pt', map_location=device, weights_only=True))

    gru_available = (DATASET_DIR_GRU / 'scaler.pkl').exists() and \
                    (MODEL_DIR / 'rally_gru_best.pt').exists()
    if gru_available:
        with open(DATASET_DIR_GRU / 'scaler.pkl', 'rb') as f:
            gru_scaler = pickle.load(f)
        gru_model = load_gru_model(device)
        print("Hand-engineered GRU loaded.")

    rows = []

    for label_stem, video_stem, surface in VAL_MATCHES:
        video_path = RAW_VIDEO_DIR / f'{video_stem}.mp4'
        seg_csv    = LABELS_DIR / f'{label_stem}_segments.csv'
        emb_path   = EMBEDDINGS_DIR / f'{video_stem}_embeddings.npy'
        feat_path  = FEATURES_DIR / f'{video_stem}_features.csv'

        print(f"\n{'='*60}")
        print(f"Match: {label_stem}  [{surface}]")

        if not video_path.exists():
            print(f"  SKIP — video not found")
            continue

        gt_segs = load_gt_segments(seg_csv)
        gt_min  = sum(e - s for s, e in gt_segs) / 60

        # Detect video fps and sample_every
        cap = cv2.VideoCapture(str(video_path))
        video_fps    = cap.get(cv2.CAP_PROP_FPS)
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration_s   = video_frames / video_fps
        sample_every = max(1, round(video_fps / 25.0))
        eff_fps      = video_fps / sample_every

        match_short = label_stem.replace('_preview', '').replace('_h264', '')

        # ── CNN GRU ───────────────────────────────────────────────────────────
        if emb_path.exists():
            print(f"  CNN GRU (sample_every={sample_every}, eff_fps={eff_fps:.0f})...")
            probs      = predict_cnn(emb_path, cnn_scaler, cnn_model, device, sample_every)
            pred_segs  = get_segments(probs, eff_fps)
            pred_min   = sum(e - s for s, e in pred_segs) / 60
            p, r, f, fc, pc, fm = compare_segments(gt_segs, pred_segs)
            print(f"    GT={len(gt_segs)} segs {gt_min:.1f}min  "
                  f"Pred={len(pred_segs)} segs {pred_min:.1f}min")
            print(f"    P={p:.3f}  R={r:.3f}  F1={f:.3f}  "
                  f"captured={fc}  partial={pc}  missed={fm}")
            rows.append(dict(
                model='cnn_gru', match=match_short, surface=surface,
                gt_segments=len(gt_segs), gt_rally_min=round(gt_min, 1),
                pred_segments=len(pred_segs), pred_rally_min=round(pred_min, 1),
                precision=round(p, 3), recall=round(r, 3), f1=round(f, 3),
                fully_captured=fc, partially_cut=pc, fully_missed=fm,
            ))
        else:
            print(f"  CNN GRU: SKIP — embeddings not found (run eval_held_out.py first)")

        # ── Hand-engineered GRU ───────────────────────────────────────────────
        if gru_available and feat_path.exists():
            print(f"  Hand-engineered GRU...")
            _, probs_g = predict_gru(feat_path, gru_scaler, gru_model, device)

            # Effective fps from feature CSV timestamps
            df_ts  = pd.read_csv(feat_path, usecols=['timestamp_s'])
            feat_fps = len(df_ts) / float(df_ts['timestamp_s'].iloc[-1])

            pred_segs_g = get_segments(probs_g, feat_fps)
            pred_min_g  = sum(e - s for s, e in pred_segs_g) / 60
            p, r, f, fc, pc, fm = compare_segments(gt_segs, pred_segs_g)
            print(f"    GT={len(gt_segs)} segs {gt_min:.1f}min  "
                  f"Pred={len(pred_segs_g)} segs {pred_min_g:.1f}min")
            print(f"    P={p:.3f}  R={r:.3f}  F1={f:.3f}  "
                  f"captured={fc}  partial={pc}  missed={fm}")
            rows.append(dict(
                model='gru', match=match_short, surface=surface,
                gt_segments=len(gt_segs), gt_rally_min=round(gt_min, 1),
                pred_segments=len(pred_segs_g), pred_rally_min=round(pred_min_g, 1),
                precision=round(p, 3), recall=round(r, 3), f1=round(f, 3),
                fully_captured=fc, partially_cut=pc, fully_missed=fm,
            ))
        elif gru_available:
            print(f"  Hand-engineered GRU: SKIP — no feature CSV")

    # ── Save ─────────────────────────────────────────────────────────────────
    if not rows:
        print("\nNo results to save.")
        return

    fieldnames = ['model', 'match', 'surface', 'gt_segments', 'gt_rally_min',
                  'pred_segments', 'pred_rally_min', 'precision', 'recall', 'f1',
                  'fully_captured', 'partially_cut', 'fully_missed']

    # Append to existing results.csv
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nAppended {len(rows)} rows to {RESULTS_CSV}")

    # Print summary table
    print(f"\n{'Model':<15} {'Match':<30} {'P':>6} {'R':>6} {'F1':>6} {'FC':>4} {'PC':>4} {'FM':>4}")
    print("-" * 80)
    for r in rows:
        print(f"{r['model']:<15} {r['match']:<30} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
              f"{r['fully_captured']:>4} {r['partially_cut']:>4} {r['fully_missed']:>4}")


if __name__ == '__main__':
    main()
