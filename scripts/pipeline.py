"""
End-to-end dead time removal pipeline.

Usage:
    python scripts/pipeline.py <video_path> [output_path] [--debug]

    python scripts/pipeline.py raw_matches/match.mp4
    python scripts/pipeline.py raw_matches/match.mp4 output/match_cut.mp4
    python scripts/pipeline.py raw_matches/match.mp4 --debug

Steps:
    1. Feature extraction  → features/<stem>_features.csv
    2. GRU inference       → per-frame rally probabilities
    3. Video cutting       → output video with dead time removed

Feature extraction is skipped if the CSV already exists.

Debug mode (--debug):
    - Saves predicted segments CSV alongside the output video
    - Keeps the converted input video (if codec conversion was needed)
"""

import sys
import pickle
import subprocess
import cv2
import torch
from pathlib import Path

ROOT         = Path(__file__).parent.parent
FEATURES_DIR = ROOT / 'features'
DATASET_DIR  = ROOT / 'gru_dataset'
MODEL_DIR    = ROOT / 'models'

sys.path.insert(0, str(ROOT / 'scripts'))
from feature_visualisation import run_feature_visualisation
from predict_video         import load_model, predict
from cut_video             import get_segments, cut_with_ffmpeg

import pandas as pd

def prepare_extraction_video(video_path):
    """
    Ensure video is readable by OpenCV for feature extraction.
    Only converts if OpenCV cannot decode the codec (e.g. AV1).
    Returns path to the prepared video (may be the original).
    """
    cap = cv2.VideoCapture(str(video_path))
    ret, _ = cap.read()
    cap.release()

    if ret:
        return video_path

    print(f"[0/3] OpenCV cannot decode {video_path.name} — converting codec ...")
    converted = video_path.parent / (video_path.stem + '_h264.mp4')
    if not converted.exists():
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(video_path),
             '-c:v', 'libx264', '-crf', '18', '-preset', 'ultrafast',
             '-c:a', 'copy', str(converted)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return converted


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/pipeline.py <video_path> [output_path]")
        sys.exit(1)

    args       = sys.argv[1:]
    debug      = '--debug' in args
    args       = [a for a in args if a != '--debug']

    video_path = Path(args[0])
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    stem     = video_path.stem
    feat_csv = FEATURES_DIR / f'{stem}_features.csv'

    if len(args) >= 2:
        out_path = Path(args[1])
    else:
        out_path = ROOT / 'runs' / 'cut' / f'{stem}_cut.mp4'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Feature extraction ─────────────────────────────────────────────
    if feat_csv.exists():
        print(f"[1/3] Feature extraction — skipping (CSV already exists)")
        extraction_video = video_path
    else:
        print(f"[1/3] Feature extraction → {feat_csv.name}")
        extraction_video = prepare_extraction_video(video_path)
        run_feature_visualisation(
            video_path=str(extraction_video),
            csv_path=str(feat_csv),
        )

    # ── Step 2: GRU inference ──────────────────────────────────────────────────
    print(f"[2/3] Running GRU inference ...", end=' ', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(DATASET_DIR / 'scaler.pkl', 'rb') as fh:
        scaler = pickle.load(fh)

    model                = load_model(device)
    frame_idx_arr, probs = predict(feat_csv, scaler, model, device)
    print(f"done  (rally={( probs >= 0.5).mean():.1%})")

    # ── Step 3: Cut video ──────────────────────────────────────────────────────
    print(f"[3/3] Cutting video ...", end=' ', flush=True)
    df       = pd.read_csv(feat_csv, usecols=['frame_idx', 'timestamp_s'])
    eff_fps  = len(frame_idx_arr) / float(df['timestamp_s'].iloc[-1])
    segments = get_segments(probs, eff_fps)
    cut_with_ffmpeg(video_path, segments, out_path)
    print("done")

    if debug:
        seg_csv = out_path.with_name(out_path.stem + '_segments.csv')
        pd.DataFrame(segments, columns=['start_s', 'end_s']).to_csv(seg_csv, index=False)
        print(f"[debug] Segments CSV: {seg_csv}")

    if extraction_video != video_path and extraction_video.exists():
        if debug:
            print(f"[debug] Keeping converted file: {extraction_video.name}")
        else:
            extraction_video.unlink()
            print(f"Removed converted file: {extraction_video.name}")

    total_in  = float(subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
        capture_output=True, text=True
    ).stdout.strip())
    total_out = sum(e - s for s, e in segments)
    print(f"\nOriginal:  {total_in/60:.1f} min")
    print(f"Cut:       {total_out/60:.1f} min  ({total_out/total_in:.0%} kept,  {len(segments)} segments)")
    print(f"Output:    {out_path}")


if __name__ == '__main__':
    main()
