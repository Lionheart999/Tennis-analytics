"""
Run GRU inference on a clip and write an annotated video showing
rally predictions vs ground truth.

Usage:
    python scripts/predict_video.py [clip_stem]

    clip_stem defaults to W2019_Djokovic_v_Federer_part_018

Output:
    runs/predict/<stem>_predicted.mp4
"""

import sys
import pickle
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path

ROOT         = Path(__file__).parent.parent
FEATURES_DIR = ROOT / 'features'
LABELS_DIR   = ROOT / 'labels'
VIDEOS_DIR   = ROOT / 'processed_videos'
DATASET_DIR  = ROOT / 'gru_dataset'
MODEL_DIR    = ROOT / 'models'
OUT_DIR      = ROOT / 'runs' / 'predict'

sys.path.insert(0, str(ROOT / 'scripts'))
from train_gru   import RallyGRU
from label_segments import expand_labels
from build_dataset  import FEATURE_COLS, WINDOW

SMOOTH_WINDOW = 5    # frames for probability smoothing
HALF          = WINDOW // 2


# ── Load model ─────────────────────────────────────────────────────────────────

def load_model(device):
    cfg = {}
    for line in (MODEL_DIR / 'rally_gru_config.txt').read_text().splitlines():
        k, v = line.split('=')
        cfg[k.strip()] = int(v.strip())
    model = RallyGRU(input_size=cfg['input_size']).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / 'rally_gru_best.pt',
                                     map_location=device, weights_only=True))
    model.eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(feat_csv, scaler, model, device):
    df = pd.read_csv(feat_csv).fillna(0.0)
    if 'p1_ax' not in df.columns:
        df['p1_ax'] = df['p1_vx'].diff().fillna(0.0)
        df['p1_ay'] = df['p1_vy'].diff().fillna(0.0)
        df['p2_ax'] = df['p2_vx'].diff().fillna(0.0)
        df['p2_ay'] = df['p2_vy'].diff().fillna(0.0)

    feat = df[FEATURE_COLS].values.astype(np.float32)
    n, f = feat.shape

    # Normalise
    feat = scaler.transform(feat)

    # Pad edges so every frame gets a centred window
    pad  = np.concatenate([feat[:HALF][::-1], feat, feat[-HALF:][::-1]])

    # Build all windows at once (n, WINDOW, f)
    windows = np.stack([pad[i:i + WINDOW] for i in range(n)], axis=0)

    batch_size = 512
    probs = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            x      = torch.tensor(windows[start:start + batch_size]).to(device)
            logits = model(x)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)

    # Smooth to reduce flicker
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    probs  = np.convolve(probs, kernel, mode='same')

    # Return aligned to frame_idx
    frame_idx = df['frame_idx'].values.astype(int)
    return frame_idx, probs


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_overlay(frame, frame_idx, probs, gt_labels, frame_idx_arr, w, h, fps):
    n_frames = len(probs)

    # Map this video frame index to prob array index
    arr_idx  = int(np.searchsorted(frame_idx_arr, frame_idx))
    arr_idx  = np.clip(arr_idx, 0, n_frames - 1)
    prob     = float(probs[arr_idx])
    is_rally = prob >= 0.5

    # ── Top banner ────────────────────────────────────────────────────────────
    banner_color = (0, 180, 60) if is_rally else (30, 30, 140)
    pred_text    = f"PRED: {'RALLY' if is_rally else 'DEAD TIME'}  {prob:.2f}"
    cv2.rectangle(frame, (0, 0), (w, 36), banner_color, -1)
    cv2.putText(frame, pred_text, (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Ground truth label in banner
    if gt_labels is not None:
        gt_fi  = np.clip(frame_idx, 0, len(gt_labels) - 1)
        gt_lbl = int(gt_labels[gt_fi])
        if gt_lbl == 1:
            gt_text, gt_color = "TRUE: RALLY",     (0, 230, 100)
        elif gt_lbl == 0:
            gt_text, gt_color = "TRUE: DEAD TIME", (100, 100, 220)
        else:
            gt_text, gt_color = "TRUE: EXCLUDED",  (180, 180, 180)
        cv2.putText(frame, gt_text, (w // 2, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, gt_color, 2)

    # Timestamp top-right
    t = frame_idx / fps
    mins, secs = divmod(t, 60)
    cv2.putText(frame, f"{int(mins):02d}:{secs:05.2f}",
                (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    # ── Timeline bars ─────────────────────────────────────────────────────────
    tl_h  = 10
    gap   = 3
    tl_y_pred = h - 18
    tl_y_gt   = tl_y_pred - tl_h - gap

    # Background
    for tl_y in (tl_y_pred, tl_y_gt):
        cv2.rectangle(frame, (0, tl_y), (w, tl_y + tl_h), (40, 40, 40), -1)

    # Prediction bar — colour by prob
    for j in range(w):
        idx  = int(j / w * n_frames)
        idx  = np.clip(idx, 0, n_frames - 1)
        p    = float(probs[idx])
        col  = (0, int(180 * p), int(60 * p)) if p >= 0.5 else (40, 40, 40)
        cv2.line(frame, (j, tl_y_pred), (j, tl_y_pred + tl_h), col, 1)

    # Ground truth bar
    if gt_labels is not None:
        for j in range(w):
            fi  = int(j / w * len(gt_labels))
            fi  = np.clip(fi, 0, len(gt_labels) - 1)
            lbl = int(gt_labels[fi])
            col = (0, 200, 80) if lbl == 1 else (40, 40, 40) if lbl == 0 else (80, 80, 80)
            cv2.line(frame, (j, tl_y_gt), (j, tl_y_gt + tl_h), col, 1)

    # Labels
    cv2.putText(frame, "pred", (2, tl_y_pred - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)
    if gt_labels is not None:
        cv2.putText(frame, "true", (2, tl_y_gt - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)

    # Playhead
    px = int(arr_idx / n_frames * w)
    cv2.line(frame, (px, tl_y_gt - 2), (px, tl_y_pred + tl_h + 2), (255, 255, 255), 2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    stem    = sys.argv[1] if len(sys.argv) > 1 else 'W2019_Djokovic_v_Federer_part_018'
    video   = VIDEOS_DIR   / f'{stem}.mp4'
    feat_csv = FEATURES_DIR / f'{stem}_features.csv'
    seg_csv  = LABELS_DIR   / f'{stem}_segments.csv'
    out_path = OUT_DIR / f'{stem}_predicted.mp4'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Clip:   {stem}")

    with open(DATASET_DIR / 'scaler.pkl', 'rb') as fh:
        scaler = pickle.load(fh)

    model = load_model(device)

    print("Running inference ...", end=' ', flush=True)
    frame_idx_arr, probs = predict(feat_csv, scaler, model, device)
    print(f"done  ({len(probs)} frames,  rally={( probs >= 0.5).mean():.1%})")

    # Ground truth
    gt_labels = None
    if seg_csv.exists():
        gt_labels, _ = expand_labels(seg_csv, video, sample_every=1)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path),
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w, h))

    fi = 0
    print("Writing annotated video ...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        draw_overlay(frame, fi, probs, gt_labels, frame_idx_arr, w, h, fps)
        out.write(frame)
        fi += 1

    cap.release()
    out.release()
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
