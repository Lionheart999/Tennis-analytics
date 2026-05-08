"""
Overlay predicted vs ground truth rally segments on a video.

Usage:
    python scripts/compare_video.py <video> <pred_csv> <gt_csv> [duration_mins]

    python scripts/compare_video.py raw_videos/match_h264.mp4 \
        runs/cut/match_cut_segments.csv \
        labels/match_segments.csv \
        10

Output:
    runs/compare/<stem>_compare.mp4
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / 'runs' / 'compare'


def build_mask(segments_df, label_col, n):
    mask = np.zeros(n, dtype=np.int8)
    for _, row in segments_df.iterrows():
        if label_col and row.get(label_col, 'rally') != 'rally':
            continue
        mask[int(row.start_s):int(row.end_s) + 1] = 1
    return mask


def draw_overlay(frame, t, pred_mask, gt_mask, w, h):
    total = len(pred_mask)
    t_i   = min(int(t), total - 1)

    is_pred = bool(pred_mask[t_i])
    is_gt   = bool(gt_mask[t_i])

    # ── Top banner ────────────────────────────────────────────────────────────
    if is_pred and is_gt:
        banner_color = (0, 160, 0)       # green  — both agree rally
        status       = 'RALLY'
    elif not is_pred and not is_gt:
        banner_color = (60, 60, 60)      # dark   — both agree dead
        status       = 'DEAD TIME'
    elif is_pred and not is_gt:
        banner_color = (0, 100, 200)     # orange — false positive
        status       = 'FALSE POSITIVE'
    else:
        banner_color = (0, 0, 180)       # red    — false negative
        status       = 'MISSED RALLY'

    cv2.rectangle(frame, (0, 0), (w, 36), banner_color, -1)
    cv2.putText(frame, f'PRED: {status}', (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Timestamp
    mins, secs = divmod(t, 60)
    cv2.putText(frame, f'{int(mins):02d}:{secs:05.2f}',
                (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    # ── Timeline bars ─────────────────────────────────────────────────────────
    tl_h      = 10
    gap       = 3
    tl_y_pred = h - 18
    tl_y_gt   = tl_y_pred - tl_h - gap

    for tl_y in (tl_y_pred, tl_y_gt):
        cv2.rectangle(frame, (0, tl_y), (w, tl_y + tl_h), (40, 40, 40), -1)

    for j in range(w):
        idx = min(int(j / w * total), total - 1)
        p_col = (0, 200, 0)   if pred_mask[idx] else (40, 40, 40)
        g_col = (0, 200, 100) if gt_mask[idx]   else (40, 40, 40)
        cv2.line(frame, (j, tl_y_pred), (j, tl_y_pred + tl_h), p_col, 1)
        cv2.line(frame, (j, tl_y_gt),   (j, tl_y_gt   + tl_h), g_col, 1)

    cv2.putText(frame, 'pred', (2, tl_y_pred - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)
    cv2.putText(frame, 'true', (2, tl_y_gt - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)

    # Playhead
    px = int(t_i / total * w)
    cv2.line(frame, (px, tl_y_gt - 2), (px, tl_y_pred + tl_h + 2), (255, 255, 255), 2)


def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/compare_video.py <video> <pred_csv> <gt_csv> [duration_mins]")
        sys.exit(1)

    video_path    = Path(sys.argv[1])
    pred_csv_path = Path(sys.argv[2])
    gt_csv_path   = Path(sys.argv[3])
    duration_mins = float(sys.argv[4]) if len(sys.argv) >= 5 else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{video_path.stem}_compare.mp4'

    pred_df = pd.read_csv(pred_csv_path)
    gt_df   = pd.read_csv(gt_csv_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps

    max_frames = int(duration_mins * 60 * fps) if duration_mins else total_frames
    n          = int(duration_s) + 1

    pred_mask = build_mask(pred_df, None,    n)
    gt_mask   = build_mask(gt_df,   'label', n)

    out = cv2.VideoWriter(str(out_path),
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w, h))

    fi = 0
    print(f"Writing comparison video ({duration_mins or duration_s/60:.1f} min) ...")
    while cap.isOpened() and fi < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        draw_overlay(frame, fi / fps, pred_mask, gt_mask, w, h)
        out.write(frame)
        fi += 1
        if fi % int(fps * 60) == 0:
            print(f"  {fi/fps/60:.1f} min processed ...")

    cap.release()
    out.release()
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
