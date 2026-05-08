"""
Train the GRU classifier on SimCLR CNN embeddings (comparison baseline).

Same architecture and hyperparameters as train_gru.py, but reads from
gru_dataset_cnn/ (512-d embeddings) instead of gru_dataset/ (34 features).

Usage:
    python scripts/train_gru_cnn.py

Saves:
    models/rally_gru_cnn_best.pt
    models/rally_gru_cnn_config.txt
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

ROOT        = Path(__file__).parent.parent
DATASET_DIR = ROOT / 'gru_dataset_cnn'
MODEL_DIR   = ROOT / 'models'

sys.path.insert(0, str(ROOT / 'scripts'))
from train_gru import RallyGRU, evaluate, HIDDEN, LAYERS, BATCH, EPOCHS, LR, PATIENCE, SEED


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    X_train = torch.tensor(np.load(DATASET_DIR / 'X_train.npy'))
    y_train = torch.tensor(np.load(DATASET_DIR / 'y_train.npy'))
    X_val   = torch.tensor(np.load(DATASET_DIR / 'X_val.npy'))
    y_val   = torch.tensor(np.load(DATASET_DIR / 'y_val.npy'))

    print(f"Train: {X_train.shape}  rally={y_train.mean():.1%}")
    print(f"Val:   {X_val.shape}    rally={y_val.mean():.1%}\n")

    pos_weight = torch.tensor(
        [(1 - y_train.mean()) / y_train.mean()]
    ).to(device)
    print(f"pos_weight: {pos_weight.item():.2f}\n")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=BATCH
    )

    model     = RallyGRU(input_size=X_train.shape[2]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_f1    = -1.0
    patience_count = 0
    history        = []

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val Acc':>7}  {'Val F1':>6}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(y_train)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(1.0 - val_f1)

        is_best = val_f1 > best_val_f1
        marker  = " ✓" if is_best else ""
        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  {val_acc:>7.3f}  {val_f1:>6.3f}{marker}")
        history.append({
            'epoch': epoch, 'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4), 'val_acc': round(val_acc, 4),
            'val_f1': round(val_f1, 4), 'best': is_best,
        })

        if is_best:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), MODEL_DIR / 'rally_gru_cnn_best.pt')
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    (MODEL_DIR / 'rally_gru_cnn_config.txt').write_text(
        f"input_size={X_train.shape[2]}\n"
        f"hidden={HIDDEN}\n"
        f"layers={LAYERS}\n"
        f"window={75}\n"
    )

    import csv
    log_path = MODEL_DIR / 'training_history_cnn.csv'
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"\nBest val F1: {best_val_f1:.4f}")
    print(f"Model saved to {MODEL_DIR / 'rally_gru_cnn_best.pt'}")
    print(f"Training log:  {log_path}")


if __name__ == '__main__':
    main()
