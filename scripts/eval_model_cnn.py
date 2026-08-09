"""
Evaluate the CNN-embedding GRU on the internal val set.

Produces:
    runs/eval/confusion_matrix_cnn.png
    runs/eval/roc_curve_cnn.png
    runs/eval/metrics_cnn.txt

Usage:
    python scripts/eval_model_cnn.py
"""

import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import sys

ROOT        = Path(__file__).parent.parent
DATASET_DIR = ROOT / 'gru_dataset_cnn'
MODEL_DIR   = ROOT / 'models'
OUT_DIR     = ROOT / 'runs' / 'eval'

sys.path.insert(0, str(ROOT / 'scripts'))
from train_gru import RallyGRU
from build_dataset_cnn import EmbeddingWindowDataset

BATCH = 512


def get_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            all_probs.append(torch.sigmoid(model(X)).cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(DATASET_DIR / 'val_index.pkl', 'rb') as f:
        val_idx = pickle.load(f)
    with open(DATASET_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    val_ds     = EmbeddingWindowDataset(val_idx, scaler)
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=4)
    y_val      = np.array([lbl for _, _, lbl in val_idx], dtype=np.float32)

    cfg = {}
    for line in (MODEL_DIR / 'rally_gru_cnn_config.txt').read_text().splitlines():
        k, v = line.split('=')
        cfg[k.strip()] = int(v.strip())

    model = RallyGRU(input_size=cfg['input_size']).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / 'rally_gru_cnn_best.pt',
                                     map_location=device, weights_only=True))

    print(f"Running inference on val set ({len(y_val):,} windows) ...")
    probs, _ = get_probs(model, val_loader, device)
    preds = (probs >= 0.5).astype(int)

    cm  = confusion_matrix(y_val, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Dead Time', 'Rally'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix — CNN GRU Val Set')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'confusion_matrix_cnn.png', dpi=150)
    plt.close(fig)
    print("Saved confusion_matrix_cnn.png")

    fpr, tpr, thresholds = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)
    idx_05  = np.argmin(np.abs(thresholds - 0.5))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f'CNN GRU (AUC = {roc_auc:.3f})')
    ax.scatter(fpr[idx_05], tpr[idx_05], s=80, zorder=5,
               label=f'Threshold = 0.5')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — CNN GRU Val Set')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'roc_curve_cnn.png', dpi=150)
    plt.close(fig)
    print("Saved roc_curve_cnn.png")

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / len(y_val)

    lines = [
        f"Val set:   {len(y_val):,} windows  (rally={y_val.mean():.1%})",
        f"AUC-ROC:   {roc_auc:.4f}",
        f"Accuracy:  {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
        f"F1:        {f1:.4f}",
        f"",
        f"Confusion matrix (threshold=0.5):",
        f"  TN={tn:,}  FP={fp:,}",
        f"  FN={fn:,}  TP={tp:,}",
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    (OUT_DIR / 'metrics_cnn.txt').write_text(text + '\n')
    print(f"\nSaved metrics_cnn.txt  →  {OUT_DIR}")


if __name__ == '__main__':
    main()
