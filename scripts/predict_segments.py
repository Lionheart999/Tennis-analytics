"""
Run GRU inference on a feature CSV and save predicted rally segments.

Usage:
    python scripts/predict_segments.py <stem>
    python scripts/predict_segments.py A2025_Sinner_v_Shelton

Output:
    runs/segments/<stem>_predicted_segments.csv  — start_s, end_s per rally
"""

import sys
import pickle
import torch
import pandas as pd
from pathlib import Path

ROOT         = Path(__file__).parent.parent
FEATURES_DIR = ROOT / 'features'
DATASET_DIR  = ROOT / 'gru_dataset'
OUT_DIR      = ROOT / 'runs' / 'segments'

sys.path.insert(0, str(ROOT / 'scripts'))
from predict_video import load_model, predict
from cut_video     import get_segments


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else None
    if not stem:
        print("Usage: python scripts/predict_segments.py <stem>")
        sys.exit(1)

    feat_csv = FEATURES_DIR / f'{stem}_features.csv'
    if not feat_csv.exists():
        print(f"ERROR: No feature CSV found: {feat_csv}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f'{stem}_predicted_segments.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(DATASET_DIR / 'scaler.pkl', 'rb') as fh:
        scaler = pickle.load(fh)

    model = load_model(device)

    print(f"Running inference on {stem} ...", end=' ', flush=True)
    frame_idx_arr, probs = predict(feat_csv, scaler, model, device)
    print(f"done  (rally={(probs >= 0.5).mean():.1%})")

    df      = pd.read_csv(feat_csv, usecols=['timestamp_s'])
    eff_fps = len(frame_idx_arr) / float(df['timestamp_s'].iloc[-1])
    segments = get_segments(probs, eff_fps)

    pd.DataFrame(segments, columns=['start_s', 'end_s']).to_csv(out_csv, index=False)
    print(f"Saved {len(segments)} segments → {out_csv}")


if __name__ == '__main__':
    main()
