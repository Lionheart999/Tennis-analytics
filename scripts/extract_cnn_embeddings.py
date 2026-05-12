"""
Extract 512-d CNN embeddings from the SimCLR-pretrained ResNet-18 encoder.

For each clip that has a feature CSV, reads the video sequentially and
extracts an embedding at every frame index that appears in the feature CSV.
Output is aligned to the feature CSV row-for-row.

Usage:
    python scripts/extract_cnn_embeddings.py

Saves per clip to embeddings/:
    <stem>_embeddings.npy   — (n_frames, 512) float32
    <stem>_frame_idx.npy    — (n_frames,)  int32 frame indices

Requires:
    models/simclr_encoder.pt  — from simclr_pretrain.py
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
import pandas as pd
from pathlib import Path

ROOT           = Path(__file__).parent.parent
VIDEO_DIR      = ROOT / 'processed_videos'
RAW_VIDEO_DIR  = ROOT / 'raw_videos'
FEATURES_DIR   = ROOT / 'features'
EMBEDDINGS_DIR = ROOT / 'embeddings'
MODEL_DIR      = ROOT / 'models'

IMG_SIZE   = 224    # must match simclr_pretrain.py
BATCH_SIZE = 64


def get_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_encoder(device):
    backbone = models.resnet18(weights=None)
    encoder  = nn.Sequential(*list(backbone.children())[:-1])   # (B, 512, 1, 1)
    sd       = torch.load(MODEL_DIR / 'simclr_encoder.pt',
                          map_location=device, weights_only=True)
    encoder.load_state_dict(sd)
    encoder.eval()
    return encoder.to(device)


def find_video(stem):
    """Look in processed_videos first, then raw_videos."""
    for d in (VIDEO_DIR, RAW_VIDEO_DIR):
        p = d / f'{stem}.mp4'
        if p.exists():
            return p
    return None


def flush_buffer(frames_buf, idx_buf, encoder, transform, device,
                 embeddings, valid_idx):
    batch = torch.stack([transform(f) for f in frames_buf]).to(device)
    with torch.no_grad():
        emb = encoder(batch).squeeze(-1).squeeze(-1)   # (B, 512)
    embeddings.append(emb.cpu().numpy())
    valid_idx.extend(idx_buf)


def extract_clip(stem, encoder, transform, device):
    feat_csv = FEATURES_DIR   / f'{stem}_features.csv'
    out_emb  = EMBEDDINGS_DIR / f'{stem}_embeddings.npy'
    out_idx  = EMBEDDINGS_DIR / f'{stem}_frame_idx.npy'

    if out_emb.exists():
        print("SKIP (exists)")
        return

    video = find_video(stem)
    if video is None:
        print("SKIP (video not found)")
        return

    df        = pd.read_csv(feat_csv)
    idx_set   = set(df['frame_idx'].astype(int).tolist())

    cap        = cv2.VideoCapture(str(video))
    embeddings = []
    valid_idx  = []
    frames_buf = []
    idx_buf    = []
    frame_num  = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_num in idx_set:
            frames_buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx_buf.append(frame_num)
            if len(frames_buf) >= BATCH_SIZE:
                flush_buffer(frames_buf, idx_buf, encoder, transform,
                             device, embeddings, valid_idx)
                frames_buf, idx_buf = [], []
        frame_num += 1

    cap.release()

    if frames_buf:
        flush_buffer(frames_buf, idx_buf, encoder, transform,
                     device, embeddings, valid_idx)

    if not embeddings:
        print("WARNING: no frames extracted")
        return

    emb_arr = np.concatenate(embeddings, axis=0).astype(np.float32)
    idx_arr = np.array(valid_idx, dtype=np.int32)
    np.save(out_emb, emb_arr)
    np.save(out_idx, idx_arr)
    print(f"{emb_arr.shape[0]} frames → {out_emb.name}")


def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    encoder   = load_encoder(device)
    transform = get_transform()

    feat_csvs = sorted(FEATURES_DIR.glob('*_features.csv'))
    print(f"Extracting embeddings for {len(feat_csvs)} clips...\n")

    for feat_csv in feat_csvs:
        stem = feat_csv.stem.replace('_features', '')
        print(f"[{stem}] ", end='', flush=True)
        extract_clip(stem, encoder, transform, device)

    print(f"\nEmbeddings saved to {EMBEDDINGS_DIR}")


if __name__ == '__main__':
    main()
