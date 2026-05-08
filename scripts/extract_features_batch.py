"""
Batch feature extraction for all labelled clips.

Usage:
    python scripts/extract_features_batch.py                        # all labelled clips
    python scripts/extract_features_batch.py raw_videos/match.mp4  # single video

For each CSV in labels/, finds the matching video in processed_videos/,
runs feature extraction, and saves to features/<stem>_features.csv.
Skips clips that already have a feature CSV.
"""

import sys
from pathlib import Path

ROOT         = Path(__file__).parent.parent
LABELS_DIR   = ROOT / 'labels'
VIDEO_DIR    = ROOT / 'processed_videos'
FEATURES_DIR = ROOT / 'features'

sys.path.insert(0, str(ROOT / 'scripts'))
from feature_visualisation import run_feature_visualisation


def extract_one(video_path):
    video_path = Path(video_path)
    feat_csv   = FEATURES_DIR / (video_path.stem + '_features.csv')
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    if feat_csv.exists():
        print(f"SKIP — features already exist: {feat_csv.name}")
        return
    print(f"Extracting features → {feat_csv.name}")
    run_feature_visualisation(video_path=str(video_path), csv_path=str(feat_csv))


def main():
    if len(sys.argv) >= 2:
        extract_one(sys.argv[1])
        return

    label_csvs = sorted(LABELS_DIR.glob('*_segments.csv'))
    if not label_csvs:
        print("No label CSVs found in", LABELS_DIR)
        return

    print(f"Found {len(label_csvs)} labelled clips\n")

    skipped = 0
    for i, label_csv in enumerate(label_csvs, 1):
        stem     = label_csv.stem.replace('_segments', '')
        video    = VIDEO_DIR / (stem + '.mp4')
        feat_csv = FEATURES_DIR / (stem + '_features.csv')

        print(f"[{i}/{len(label_csvs)}] {stem}")

        if not video.exists():
            print(f"  SKIP — video not found: {video}\n")
            skipped += 1
            continue

        if feat_csv.exists():
            print(f"  SKIP — features already extracted\n")
            skipped += 1
            continue

        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        run_feature_visualisation(video_path=str(video), csv_path=str(feat_csv))
        print()

    done = len(label_csvs) - skipped
    print(f"\nDone. {done} extracted, {skipped} skipped.")
    print(f"Feature CSVs in: {FEATURES_DIR}")


if __name__ == '__main__':
    main()
