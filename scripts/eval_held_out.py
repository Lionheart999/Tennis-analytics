"""
Evaluate both GRU models on the three held-out test matches.

CNN model:           auto-extracts embeddings if not already present.
Hand-engineered GRU: requires feature CSVs in features/ — if absent,
                     that model is skipped with instructions printed.

Val matches:
    A2025_Sinner_v_Shelton   (labels/val/A2025_Sinner_v_Shelton_preview_segments.csv)
    R2025_Musetti_v_Tiafoe   (labels/val/R2025_Musetti_v_Tiafoe_h264_segments.csv)
    W2019_Federer_v_Nadal    (labels/val/W2019_Federer_v_Nadal_h264_segments.csv)

Usage:
    python scripts/eval_held_out.py

Saves:
    runs/eval/held_out_metrics.txt
"""

import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

ROOT          = Path(__file__).parent.parent
LABELS_DIR    = ROOT / 'labels' / 'val'
RAW_VIDEO_DIR = ROOT / 'raw_videos'
EMBEDDINGS_DIR= ROOT / 'embeddings'
FEATURES_DIR  = ROOT / 'features'
MODEL_DIR     = ROOT / 'models'
DATASET_DIR_CNN = ROOT / 'gru_dataset_cnn'
DATASET_DIR_GRU = ROOT / 'gru_dataset'
OUT_DIR       = ROOT / 'runs' / 'eval'

sys.path.insert(0, str(ROOT / 'scripts'))
from label_segments import expand_labels
from train_gru import RallyGRU
from build_dataset_cnn import EmbeddingWindowDataset, WINDOW

# label_stem → (video_stem, video_dir)
VAL_MATCHES = [
    ('A2025_Sinner_v_Shelton_preview', 'A2025_Sinner_v_Shelton_h264', RAW_VIDEO_DIR),
    ('R2025_Musetti_v_Tiafoe_h264',    'R2025_Musetti_v_Tiafoe_h264', RAW_VIDEO_DIR),
    ('W2019_Federer_v_Nadal_h264',     'W2019_Federer_v_Nadal_h264',  RAW_VIDEO_DIR),
]

IMG_SIZE   = 224
BATCH_SIZE = 512
STRIDE_EVAL = 1    # stride=1 at eval → prediction for every frame centre


# ── CNN embedding extraction ────────────────────────────────────────────────

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


def extract_embeddings(video_path, out_path, encoder, transform, device):
    cap        = cv2.VideoCapture(str(video_path))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    embeddings = []
    buf        = []
    done       = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(buf) >= 64:
            batch = torch.stack([transform(f) for f in buf]).to(device)
            with torch.no_grad():
                emb = encoder(batch).squeeze(-1).squeeze(-1)
            embeddings.append(emb.cpu().numpy())
            done += len(buf)
            buf  = []
            if done % 10000 == 0:
                print(f"    {done:,}/{total:,} frames", end='\r', flush=True)

    cap.release()
    if buf:
        batch = torch.stack([transform(f) for f in buf]).to(device)
        with torch.no_grad():
            emb = encoder(batch).squeeze(-1).squeeze(-1)
        embeddings.append(emb.cpu().numpy())

    arr = np.concatenate(embeddings, axis=0).astype(np.float32)
    np.save(out_path, arr)
    print(f"    {arr.shape[0]:,} frames → {out_path.name}        ")
    return arr


# ── Inference helpers ────────────────────────────────────────────────────────

def run_inference_cnn(emb_path, labels, scaler, model, device, sample_every=1):
    """Returns (probs, y_true) arrays for a single clip.

    sample_every: subsample embeddings to match training fps (e.g. 2 for 50fps→25fps).
    Transforms the full embedding array once for speed.
    """
    emb_full   = np.load(emb_path, mmap_mode='r')
    emb_sub    = emb_full[::sample_every]           # strided view
    emb_scaled = scaler.transform(emb_sub).astype(np.float32)  # transform once
    n          = len(emb_scaled)

    starts, y_true = [], []
    for start in range(0, n - WINDOW + 1, STRIDE_EVAL):
        center = start + WINDOW // 2
        lbl    = int(labels[center]) if center < len(labels) else 0
        if lbl == -1:
            continue
        starts.append(start)
        y_true.append(lbl)

    if not starts:
        return np.array([]), np.array([])

    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), BATCH_SIZE):
            batch = starts[i:i + BATCH_SIZE]
            X = np.stack([emb_scaled[s:s + WINDOW] for s in batch])
            X = torch.tensor(X).to(device)
            all_probs.append(torch.sigmoid(model(X)).cpu().numpy())

    return np.concatenate(all_probs), np.array(y_true, dtype=np.float32)


def run_inference_gru(feat_path, labels, scaler, model, device):
    """Returns (probs, y_true) for hand-engineered GRU on one clip.

    labels must already be generated with the correct sample_every for this clip.
    """
    import pandas as pd
    from build_dataset import FEATURE_COLS, WINDOW as WIN_GRU

    df    = pd.read_csv(feat_path)
    feats = scaler.transform(
        df[FEATURE_COLS].fillna(0).values.astype(np.float32)
    ).astype(np.float32)
    n = len(feats)

    index, y_true = [], []
    for start in range(0, n - WIN_GRU + 1, STRIDE_EVAL):
        center = start + WIN_GRU // 2
        lbl    = int(labels[center]) if center < len(labels) else 0
        if lbl == -1:
            continue
        index.append(start)
        y_true.append(lbl)

    if not index:
        return np.array([]), np.array([])

    # Batch the windows
    X = np.stack([feats[s:s + WIN_GRU] for s in index])
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            x = torch.tensor(X[i:i + BATCH_SIZE]).to(device)
            all_probs.append(torch.sigmoid(model(x)).cpu().numpy())

    return np.concatenate(all_probs), np.array(y_true, dtype=np.float32)


# ── Metrics reporting ────────────────────────────────────────────────────────

def compute_metrics(probs, y_true, name):
    preds = (probs >= 0.5).astype(int)
    cm    = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / len(y_true)
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc   = auc(fpr, tpr)
    return dict(name=name, n=len(y_true), rally_pct=y_true.mean(),
                auc=roc_auc, acc=accuracy, prec=precision,
                rec=recall, f1=f1, cm=cm)


def print_metrics(m):
    print(f"  Windows: {m['n']:,}  rally={m['rally_pct']:.1%}")
    print(f"  AUC={m['auc']:.4f}  Acc={m['acc']:.4f}  "
          f"P={m['prec']:.4f}  R={m['rec']:.4f}  F1={m['f1']:.4f}")
    tn, fp, fn, tp = m['cm'].ravel()
    print(f"  TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")


def save_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Dead Time', 'Rally']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Load CNN GRU ──────────────────────────────────────────────────────────
    with open(DATASET_DIR_CNN / 'scaler.pkl', 'rb') as f:
        cnn_scaler = pickle.load(f)
    cnn_model = RallyGRU(input_size=512).to(device)
    cnn_model.load_state_dict(torch.load(
        MODEL_DIR / 'rally_gru_cnn_best.pt', map_location=device, weights_only=True))

    # ── Load hand-engineered GRU (if available) ───────────────────────────────
    gru_available = (DATASET_DIR_GRU / 'scaler.pkl').exists() and \
                    (MODEL_DIR / 'rally_gru_best.pt').exists()
    if gru_available:
        with open(DATASET_DIR_GRU / 'scaler.pkl', 'rb') as f:
            gru_scaler = pickle.load(f)
        cfg = {}
        for line in (MODEL_DIR / 'rally_gru_config.txt').read_text().splitlines():
            k, v = line.split('=')
            cfg[k.strip()] = int(v.strip())
        gru_model = RallyGRU(input_size=cfg['input_size']).to(device)
        gru_model.load_state_dict(torch.load(
            MODEL_DIR / 'rally_gru_best.pt', map_location=device, weights_only=True))
        print("Hand-engineered GRU loaded.")
    else:
        print("Hand-engineered GRU skipped (scaler or model not found).")

    # ── Load SimCLR encoder for embedding extraction ──────────────────────────
    encoder   = load_encoder(device)
    transform = get_transform()

    cnn_all_probs, cnn_all_labels = [], []
    gru_all_probs, gru_all_labels = [], []

    report_lines = ["Held-out test set evaluation", "=" * 60, ""]

    for label_stem, video_stem, video_dir in VAL_MATCHES:
        video_path = video_dir / f'{video_stem}.mp4'
        seg_path   = LABELS_DIR / f'{label_stem}_segments.csv'
        emb_path   = EMBEDDINGS_DIR / f'{video_stem}_embeddings.npy'

        print(f"\n{'='*60}")
        print(f"Match: {label_stem}")

        if not video_path.exists():
            print(f"  SKIP — video not found: {video_path}")
            continue

        # Extract embeddings if needed
        if not emb_path.exists():
            print(f"  Extracting CNN embeddings...")
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            extract_embeddings(video_path, emb_path, encoder, transform, device)

        # Detect fps ratio so both models evaluate at 25fps (training fps).
        # e.g. 50fps raw video with 25fps feature CSV → sample_every=2
        cap = cv2.VideoCapture(str(video_path))
        video_fps    = cap.get(cv2.CAP_PROP_FPS)
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration_s   = video_frames / video_fps if video_fps > 0 else 1
        emb_n        = np.load(emb_path, mmap_mode='r').shape[0]
        emb_fps      = emb_n / duration_s
        sample_every = max(1, round(video_fps / 25.0))   # normalise to 25fps
        print(f"  Video: {video_fps:.0f}fps  embeddings: {emb_fps:.0f}fps  "
              f"sample_every={sample_every}")

        # Labels at 25fps (one label per subsampled frame)
        labels, _ = expand_labels(seg_path, video_path, sample_every=sample_every)

        # CNN GRU
        print(f"  Running CNN GRU inference (stride={STRIDE_EVAL})...")
        probs, y = run_inference_cnn(emb_path, labels, cnn_scaler, cnn_model, device,
                                     sample_every=sample_every)
        if len(probs):
            m = compute_metrics(probs, y, label_stem)
            print(f"  [CNN GRU]")
            print_metrics(m)
            cnn_all_probs.append(probs)
            cnn_all_labels.append(y)
            report_lines += [f"{label_stem} — CNN GRU",
                             f"  n={m['n']:,} rally={m['rally_pct']:.1%} "
                             f"AUC={m['auc']:.4f} F1={m['f1']:.4f}", ""]

        # Hand-engineered GRU
        if gru_available:
            feat_path = FEATURES_DIR / f'{video_stem}_features.csv'
            if feat_path.exists():
                print(f"  Running hand-engineered GRU inference...")
                probs_g, y_g = run_inference_gru(
                    feat_path, labels, gru_scaler, gru_model, device)
                if len(probs_g):
                    m_g = compute_metrics(probs_g, y_g, label_stem)
                    print(f"  [Hand-engineered GRU]")
                    print_metrics(m_g)
                    gru_all_probs.append(probs_g)
                    gru_all_labels.append(y_g)
                    report_lines += [f"{label_stem} — Hand-engineered GRU",
                                     f"  n={m_g['n']:,} rally={m_g['rally_pct']:.1%} "
                                     f"AUC={m_g['auc']:.4f} F1={m_g['f1']:.4f}", ""]
            else:
                print(f"  [Hand-engineered GRU] SKIP — no feature CSV found.")
                print(f"    To extract: python scripts/extract_features_batch.py {video_path}")

    # ── Aggregate results ─────────────────────────────────────────────────────
    report_lines += ["=" * 60, "AGGREGATE (all held-out matches)", ""]

    if cnn_all_probs:
        probs_agg = np.concatenate(cnn_all_probs)
        y_agg     = np.concatenate(cnn_all_labels)
        m = compute_metrics(probs_agg, y_agg, 'aggregate')
        print(f"\n{'='*60}")
        print("AGGREGATE — CNN GRU")
        print_metrics(m)
        save_confusion(m['cm'], 'Held-out Set — CNN GRU',
                       OUT_DIR / 'held_out_confusion_cnn.png')
        report_lines += [
            "CNN GRU (SimCLR embeddings):",
            f"  Windows:   {m['n']:,}  rally={m['rally_pct']:.1%}",
            f"  AUC-ROC:   {m['auc']:.4f}",
            f"  Accuracy:  {m['acc']:.4f}",
            f"  Precision: {m['prec']:.4f}",
            f"  Recall:    {m['rec']:.4f}",
            f"  F1:        {m['f1']:.4f}",
            "", "",
        ]

    if gru_all_probs:
        probs_agg = np.concatenate(gru_all_probs)
        y_agg     = np.concatenate(gru_all_labels)
        m = compute_metrics(probs_agg, y_agg, 'aggregate')
        print(f"\nAGGREGATE — Hand-engineered GRU")
        print_metrics(m)
        save_confusion(m['cm'], 'Held-out Set — Hand-engineered GRU',
                       OUT_DIR / 'held_out_confusion_gru.png')
        report_lines += [
            "Hand-engineered GRU (34 features):",
            f"  Windows:   {m['n']:,}  rally={m['rally_pct']:.1%}",
            f"  AUC-ROC:   {m['auc']:.4f}",
            f"  Accuracy:  {m['acc']:.4f}",
            f"  Precision: {m['prec']:.4f}",
            f"  Recall:    {m['rec']:.4f}",
            f"  F1:        {m['f1']:.4f}",
        ]
    elif gru_available:
        report_lines += ["Hand-engineered GRU: no feature CSVs found for any val match.",
                         "Run extract_features_batch.py on each val video first."]

    text = '\n'.join(report_lines)
    (OUT_DIR / 'held_out_metrics.txt').write_text(text + '\n')
    print(f"\nSaved to {OUT_DIR / 'held_out_metrics.txt'}")


if __name__ == '__main__':
    main()
