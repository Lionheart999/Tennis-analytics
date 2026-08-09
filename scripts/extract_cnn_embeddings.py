"""
Extract 512-d CNN embeddings from the SimCLR-pretrained ResNet-18 encoder.

Reads every frame from each processed video clip sequentially and extracts
a 512-d embedding. Frame index = position in the array, so labels from
expand_labels() align directly without any extra mapping.

Usage:
    python scripts/extract_cnn_embeddings.py

Saves per clip to embeddings/:
    <stem>_embeddings.npy   — (n_frames, 512) float32

Requires:
    models/simclr_encoder.pt  — from simclr_pretrain.py
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
from pathlib import Path

ROOT           = Path(__file__).parent.parent
VIDEO_DIR      = ROOT / 'processed_videos'
RAW_VIDEO_DIR  = ROOT / 'raw_videos' / 'Converted'
LABELS_DIR     = ROOT / 'labels'
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
    encoder  = nn.Sequential(*list(backbone.children())[:-1])
    sd       = torch.load(MODEL_DIR / 'simclr_encoder.pt',
                          map_location=device, weights_only=True)
    encoder.load_state_dict(sd)
    encoder.eval()
    return encoder.to(device)


def find_video(stem):
    for d in (VIDEO_DIR, RAW_VIDEO_DIR):
        p = d / f'{stem}.mp4'
        if p.exists():
            return p
    return None


def extract_clip(stem, encoder, transform, device):
    out_emb = EMBEDDINGS_DIR / f'{stem}_embeddings.npy'

    if out_emb.exists():
        print("SKIP (exists)")
        return

    video = find_video(stem)
    if video is None:
        print("SKIP (video not found)")
        return

    cap        = cv2.VideoCapture(str(video))
    embeddings = []
    frames_buf = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames_buf) >= BATCH_SIZE:
            batch = torch.stack([transform(f) for f in frames_buf]).to(device)
            with torch.no_grad():
                emb = encoder(batch).squeeze(-1).squeeze(-1)
            embeddings.append(emb.cpu().numpy())
            frames_buf = []

    cap.release()

    if frames_buf:
        batch = torch.stack([transform(f) for f in frames_buf]).to(device)
        with torch.no_grad():
            emb = encoder(batch).squeeze(-1).squeeze(-1)
        embeddings.append(emb.cpu().numpy())

    if not embeddings:
        print("WARNING: no frames extracted")
        return

    emb_arr = np.concatenate(embeddings, axis=0).astype(np.float32)
    np.save(out_emb, emb_arr)
    print(f"{emb_arr.shape[0]} frames → {out_emb.name}")


def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    stems = sorted(
        s.stem.replace('_segments', '')
        for s in LABELS_DIR.glob('*_segments.csv')
    )
    print(f"Extracting embeddings for {len(stems)} clips...\n")

    encoder   = load_encoder(device)
    transform = get_transform()

    for stem in stems:
        print(f"[{stem}] ", end='', flush=True)
        extract_clip(stem, encoder, transform, device)

    print(f"\nEmbeddings saved to {EMBEDDINGS_DIR}")


if __name__ == '__main__':
    main()
