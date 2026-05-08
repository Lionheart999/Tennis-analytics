"""
Feature ablation study — group-level and individual-level.

Zeros out one feature (or feature group) at a time, retrains the GRU
from scratch, and reports val F1 drop vs the baseline saved model.

Usage:
    python scripts/ablation.py --mode group        # 5 runs  (~20 min)
    python scripts/ablation.py --mode individual   # 34 runs (~2-3 hrs)
    python scripts/ablation.py --mode both         # 39 runs

Output:
    runs/eval/ablation_group_results.csv   + ablation_group_chart.png
    runs/eval/ablation_individual_results.csv + ablation_individual_chart.png
"""

import sys
import csv
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DATASET_DIR = ROOT / 'gru_dataset'
MODEL_DIR   = ROOT / 'models'
OUT_DIR     = ROOT / 'runs' / 'eval'

sys.path.insert(0, str(ROOT / 'scripts'))
from train_gru import RallyGRU
from build_dataset import FEATURE_COLS

# ── Feature group definitions ─────────────────────────────────────────────────
GROUPS = {
    'ball': [
        'ball_detected', 'ball_conf', 'ball_speed',
        'ball_freq_2s', 'ball_freq_5s', 'ball_streak_norm', 'time_since_ball_s',
    ],
    'player_mag': [
        'p1_speed', 'p2_speed', 'combined_speed',
        'p1_accel', 'p2_accel',
    ],
    'player_dir': [
        'p1_vx', 'p1_vy', 'p2_vx', 'p2_vy',
        'p1_ax', 'p1_ay', 'p2_ax', 'p2_ay',
    ],
    'player_roll': [
        'p1_move_intensity', 'p2_move_intensity',
        'p1_avg_speed_3s', 'p2_avg_speed_3s',
        'p1_max_speed_3s', 'p2_max_speed_3s',
        'p1_std_speed_3s', 'p2_std_speed_3s',
        'time_since_move_s',
    ],
    'court': [
        'both_players_visible', 'zoom_level', 'separation',
        'frame_diff', 'camera_cut',
    ],
}

COL_IDX = {name: i for i, name in enumerate(FEATURE_COLS)}

SEED = 42

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

# ── Hyperparameters (mirror train_gru.py) ─────────────────────────────────────
HIDDEN   = 64
LAYERS   = 2
DROPOUT  = 0.3
BATCH    = 256
EPOCHS   = 50
LR       = 1e-3
PATIENCE = 7


def get_baseline_f1(X_val, y_val, device):
    cfg = {}
    for line in (MODEL_DIR / 'rally_gru_config.txt').read_text().splitlines():
        k, v = line.split('=')
        cfg[k.strip()] = int(v.strip())
    model = RallyGRU(input_size=cfg['input_size']).to(device)
    model.load_state_dict(torch.load(
        MODEL_DIR / 'rally_gru_best.pt', map_location=device, weights_only=True
    ))
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for X, y in DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH):
            X, y  = X.to(device), y.to(device)
            preds = (torch.sigmoid(model(X)) > 0.5).float()
            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()
    return tp / (tp + 0.5 * (fp + fn) + 1e-9)


def train_and_eval(X_train, y_train, X_val, y_val, device, label=''):
    pos_weight = torch.tensor(
        [(1 - y_train.mean()) / (y_train.mean() + 1e-9)]
    ).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH,
    )
    set_seed()
    model     = RallyGRU(input_size=X_train.shape[2]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_f1 = -1.0
    patience_count = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y  = X.to(device), y.to(device)
                preds = (torch.sigmoid(model(X)) > 0.5).float()
                tp += ((preds == 1) & (y == 1)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()
        f1 = tp / (tp + 0.5 * (fp + fn) + 1e-9)
        scheduler.step(1.0 - f1)

        if f1 > best_f1:
            best_f1        = f1
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break
        print(f"  [{label}]  epoch {epoch:>2}  val F1={f1:.4f}{'  ✓' if f1 == best_f1 else ''}")

    return best_f1


def zero_cols(X, col_indices):
    X = X.copy()
    for idx in col_indices:
        X[:, :, idx] = 0.0
    return X


def run_ablation(X_train, y_train, X_val, y_val, device, baseline_f1,
                 items, csv_name, chart_name, total_label):
    """
    items: list of (label, col_indices) tuples
    """
    results = [{'feature': 'baseline', 'val_f1': round(baseline_f1, 4), 'f1_drop': 0.0}]
    n = len(items)
    for i, (label, col_indices) in enumerate(items, 1):
        print(f"\n{'='*55}")
        print(f"[{i}/{n}] Ablating: {label}  ({len(col_indices)} feature(s) zeroed)")
        print('='*55)
        Xtr = zero_cols(X_train, col_indices)
        Xvl = zero_cols(X_val,   col_indices)
        f1  = train_and_eval(Xtr, y_train, Xvl, y_val, device, label=label)
        drop = baseline_f1 - f1
        results.append({'feature': label, 'val_f1': round(f1, 4), 'f1_drop': round(drop, 4)})
        print(f"  → F1={f1:.4f}  drop={drop:+.4f}")

    # Save CSV
    csv_path = OUT_DIR / csv_name
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {csv_path}")

    # Bar chart
    rows   = [r for r in results if r['feature'] != 'baseline']
    labels = [r['feature'] for r in rows]
    drops  = [r['f1_drop'] for r in rows]
    colors = ['#d62728' if d > 0 else '#2ca02c' for d in drops]

    fig_w  = max(7, len(labels) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    bars = ax.bar(labels, drops, color=colors, width=0.6)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('F1 Drop (baseline − ablated)')
    ax.set_title(f'{total_label} Ablation  (baseline F1={baseline_f1:.3f})')
    ax.tick_params(axis='x', rotation=30)
    for bar, d in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                f'{d:+.3f}', ha='center', va='bottom', fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / chart_name, dpi=150)
    plt.close(fig)
    print(f"Saved {chart_name}")

    # Summary
    print(f"\n{'Feature':<22}  {'Val F1':>7}  {'Drop':>7}")
    print("-" * 38)
    for r in results:
        sign = f"{r['f1_drop']:+.4f}" if r['feature'] != 'baseline' else "—"
        print(f"{r['feature']:<22}  {r['val_f1']:>7.4f}  {sign:>7}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['group', 'individual', 'both'],
                        default='both')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X_train = np.load(DATASET_DIR / 'X_train.npy')
    y_train = np.load(DATASET_DIR / 'y_train.npy')
    X_val   = np.load(DATASET_DIR / 'X_val.npy')
    y_val   = np.load(DATASET_DIR / 'y_val.npy')

    print("Evaluating baseline (saved model) ...")
    baseline_f1 = get_baseline_f1(X_val, y_val, device)
    print(f"Baseline val F1: {baseline_f1:.4f}\n")

    if args.mode in ('group', 'both'):
        group_items = [
            (name, [COL_IDX[f] for f in feats])
            for name, feats in GROUPS.items()
        ]
        run_ablation(X_train, y_train, X_val, y_val, device, baseline_f1,
                     items=group_items,
                     csv_name='ablation_group_results.csv',
                     chart_name='ablation_group_chart.png',
                     total_label='Feature Group')

    if args.mode in ('individual', 'both'):
        individual_items = [
            (feat, [COL_IDX[feat]])
            for feat in FEATURE_COLS
        ]
        run_ablation(X_train, y_train, X_val, y_val, device, baseline_f1,
                     items=individual_items,
                     csv_name='ablation_individual_results.csv',
                     chart_name='ablation_individual_chart.png',
                     total_label='Individual Feature')


if __name__ == '__main__':
    main()
