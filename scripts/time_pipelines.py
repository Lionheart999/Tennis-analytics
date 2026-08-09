"""
End-to-end pipeline timing comparison: hand-engineered vs SimCLR.

Runs both pipelines on a held-out validation video from scratch, times each
step, computes segment-level metrics against ground truth, and prints a
side-by-side summary table.

Usage:
    python scripts/time_pipelines.py <video_path>

    e.g. python scripts/time_pipelines.py raw_videos/W2019_Federer_v_Nadal_h264.mp4

Output:
    runs/cut/<stem>_he_cut.mp4       — hand-engineered output
    runs/cut/<stem>_simclr_cut.mp4   — SimCLR output
"""

import sys
import time
import pickle
import tempfile
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
import pandas as pd
from pathlib import Path

ROOT         = Path(__file__).parent.parent
MODEL_DIR    = ROOT / 'models'
DATASET_DIR  = ROOT / 'gru_dataset'
CNN_DIR      = ROOT / 'gru_dataset_cnn'
LABELS_DIR   = ROOT / 'labels' / 'val'
OUT_DIR      = ROOT / 'runs' / 'cut'

sys.path.insert(0, str(ROOT / 'scripts'))
from feature_visualisation  import run_feature_visualisation
from predict_video          import load_model as load_gru, predict as predict_gru
from predict_video          import SMOOTH_WINDOW, HALF
from cut_video              import get_segments, cut_with_ffmpeg
from train_gru              import RallyGRU
from build_dataset          import WINDOW
from eval_segments_held_out import load_gt_segments, compare_segments

IMG_SIZE   = 224
BATCH_SIZE = 64

LABEL_MAP = {
    'A2025_Sinner_v_Shelton_h264': 'A2025_Sinner_v_Shelton_preview',
    'A2025_Sinner_v_Shelton':      'A2025_Sinner_v_Shelton_preview',
    'R2025_Musetti_v_Tiafoe_h264': 'R2025_Musetti_v_Tiafoe_h264',
    'R2025_Musetti_v_Tiafoe':      'R2025_Musetti_v_Tiafoe_h264',
    'W2019_Federer_v_Nadal_h264':  'W2019_Federer_v_Nadal_h264',
    'W2019_Federer_v_Nadal':       'W2019_Federer_v_Nadal_h264',
}


def fmt(secs):
    if secs >= 60:
        return f"{secs / 60:.1f} min"
    return f"{secs:.1f} s"


def video_duration(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


# ── SimCLR helpers ────────────────────────────────────────────────────────────

def load_encoder(device):
    backbone = models.resnet18(weights=None)
    encoder  = nn.Sequential(*list(backbone.children())[:-1])
    sd = torch.load(MODEL_DIR / 'simclr_encoder.pt',
                    map_location=device, weights_only=True)
    encoder.load_state_dict(sd)
    encoder.eval()
    return encoder.to(device)


def load_cnn_gru(device):
    model = RallyGRU(input_size=512).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / 'rally_gru_cnn_best.pt',
                                     map_location=device, weights_only=True))
    model.eval()
    return model


def extract_embeddings(video_path, encoder, device):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap  = cv2.VideoCapture(str(video_path))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    embs = []
    buf  = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(buf) >= BATCH_SIZE:
            batch = torch.stack([transform(f) for f in buf]).to(device)
            with torch.no_grad():
                embs.append(encoder(batch).squeeze(-1).squeeze(-1).cpu().numpy())
            buf = []

    cap.release()
    if buf:
        batch = torch.stack([transform(f) for f in buf]).to(device)
        with torch.no_grad():
            embs.append(encoder(batch).squeeze(-1).squeeze(-1).cpu().numpy())

    return np.concatenate(embs, axis=0).astype(np.float32), fps


def predict_from_embeddings(embeddings, scaler, model, device):
    feat = scaler.transform(embeddings).astype(np.float32)
    n    = len(feat)
    pad  = np.concatenate([feat[:HALF][::-1], feat, feat[-HALF:][::-1]])

    probs = []
    with torch.no_grad():
        for start in range(0, n, 512):
            end   = min(start + 512, n)
            batch = np.stack([pad[i:i + WINDOW] for i in range(start, end)])
            x     = torch.tensor(batch).to(device)
            probs.append(torch.sigmoid(model(x)).cpu().numpy())

    probs  = np.concatenate(probs)
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    return np.convolve(probs, kernel, mode='same')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/time_pipelines.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    stem       = video_path.stem
    label_stem = LABEL_MAP.get(stem)
    seg_csv    = LABELS_DIR / f'{label_stem}_segments.csv' if label_stem else None
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    duration   = video_duration(video_path)

    print(f"Device:  {device}")
    print(f"Video:   {video_path.name}  ({duration / 60:.1f} min)\n")

    gt_segs = load_gt_segments(seg_csv) if (seg_csv and seg_csv.exists()) else None
    if gt_segs:
        gt_min = sum(e - s for s, e in gt_segs) / 60
        print(f"Ground truth: {len(gt_segs)} segments  {gt_min:.1f} min rally\n")
    else:
        print("WARNING: no ground truth labels found — metrics will be skipped\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        feat_csv = Path(tmp) / f'{stem}_features.csv'

        # ── Hand-engineered pipeline ─────────────────────────────────────────
        print("━" * 54)
        print("HAND-ENGINEERED PIPELINE")
        print("━" * 54)

        t0 = time.perf_counter()
        print("[1/3] Feature extraction (YOLO) ...", end=' ', flush=True)
        run_feature_visualisation(video_path=str(video_path), csv_path=str(feat_csv))
        t_feat = time.perf_counter() - t0
        print(fmt(t_feat))

        t0 = time.perf_counter()
        print("[2/3] GRU inference ...", end=' ', flush=True)
        with open(DATASET_DIR / 'scaler.pkl', 'rb') as fh:
            he_scaler = pickle.load(fh)
        he_model = load_gru(device)
        _, he_probs = predict_gru(feat_csv, he_scaler, he_model, device)
        t_inf_he = time.perf_counter() - t0
        print(fmt(t_inf_he))

        t0 = time.perf_counter()
        print("[3/3] Video cutting (FFmpeg) ...", end=' ', flush=True)
        df_ts   = pd.read_csv(feat_csv, usecols=['timestamp_s'])
        eff_fps = len(he_probs) / float(df_ts['timestamp_s'].iloc[-1])
        he_segs = get_segments(he_probs, eff_fps)
        out_he  = OUT_DIR / f'{stem}_he_cut.mp4'
        cut_with_ffmpeg(video_path, he_segs, out_he)
        t_cut_he = time.perf_counter() - t0
        print(fmt(t_cut_he))

        t_total_he  = t_feat + t_inf_he + t_cut_he
        he_pred_min = sum(e - s for s, e in he_segs) / 60
        print(f"  → {len(he_segs)} segments  {he_pred_min:.1f} min kept")

        if gt_segs:
            p, r, f, fc, pc, fm = compare_segments(gt_segs, he_segs)
            print(f"  → P={p:.3f}  R={r:.3f}  F1={f:.3f}  "
                  f"FC={fc}  PC={pc}  FM={fm}")
            he_metrics = (p, r, f, fc, pc, fm)
        else:
            he_metrics = None

        # ── SimCLR pipeline ──────────────────────────────────────────────────
        print()
        print("━" * 54)
        print("SIMCLR PIPELINE")
        print("━" * 54)

        t0 = time.perf_counter()
        print("[1/3] Embedding extraction (ResNet-18) ...", end=' ', flush=True)
        encoder    = load_encoder(device)
        embeddings, fps = extract_embeddings(video_path, encoder, device)
        t_emb = time.perf_counter() - t0
        print(fmt(t_emb))

        t0 = time.perf_counter()
        print("[2/3] GRU inference ...", end=' ', flush=True)
        with open(CNN_DIR / 'scaler.pkl', 'rb') as fh:
            cnn_scaler = pickle.load(fh)
        cnn_model  = load_cnn_gru(device)
        cnn_probs  = predict_from_embeddings(embeddings, cnn_scaler, cnn_model, device)
        t_inf_cnn  = time.perf_counter() - t0
        print(fmt(t_inf_cnn))

        t0 = time.perf_counter()
        print("[3/3] Video cutting (FFmpeg) ...", end=' ', flush=True)
        cnn_segs = get_segments(cnn_probs, fps)
        out_cnn  = OUT_DIR / f'{stem}_simclr_cut.mp4'
        cut_with_ffmpeg(video_path, cnn_segs, out_cnn)
        t_cut_cnn = time.perf_counter() - t0
        print(fmt(t_cut_cnn))

        t_total_cnn  = t_emb + t_inf_cnn + t_cut_cnn
        cnn_pred_min = sum(e - s for s, e in cnn_segs) / 60
        print(f"  → {len(cnn_segs)} segments  {cnn_pred_min:.1f} min kept")

        if gt_segs:
            p, r, f, fc, pc, fm = compare_segments(gt_segs, cnn_segs)
            print(f"  → P={p:.3f}  R={r:.3f}  F1={f:.3f}  "
                  f"FC={fc}  PC={pc}  FM={fm}")
            cnn_metrics = (p, r, f, fc, pc, fm)
        else:
            cnn_metrics = None

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 36
    print()
    print("━" * (W + 20))
    print(f"{'':>{W}} {'Hand-eng':>9} {'SimCLR':>9}")
    print("─" * (W + 20))
    print(f"{'Feature / embedding extraction':<{W}} {fmt(t_feat):>9} {fmt(t_emb):>9}")
    print(f"{'GRU inference':<{W}} {fmt(t_inf_he):>9} {fmt(t_inf_cnn):>9}")
    print(f"{'Video cutting (FFmpeg)':<{W}} {fmt(t_cut_he):>9} {fmt(t_cut_cnn):>9}")
    print("─" * (W + 20))
    print(f"{'Total':<{W}} {fmt(t_total_he):>9} {fmt(t_total_cnn):>9}")
    print(f"{'Realtime factor  (total ÷ duration)':<{W}} {t_total_he/duration:>8.1f}x {t_total_cnn/duration:>8.1f}x")
    if he_metrics and cnn_metrics:
        print("─" * (W + 20))
        print(f"{'Precision':<{W}} {he_metrics[0]:>9.3f} {cnn_metrics[0]:>9.3f}")
        print(f"{'Recall':<{W}} {he_metrics[1]:>9.3f} {cnn_metrics[1]:>9.3f}")
        print(f"{'F1':<{W}} {he_metrics[2]:>9.3f} {cnn_metrics[2]:>9.3f}")
        print(f"{'Fully captured':<{W}} {he_metrics[3]:>9} {cnn_metrics[3]:>9}")
        print(f"{'Partially cut':<{W}} {he_metrics[4]:>9} {cnn_metrics[4]:>9}")
        print(f"{'Fully missed':<{W}} {he_metrics[5]:>9} {cnn_metrics[5]:>9}")
    print()
    print(f"Outputs saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
