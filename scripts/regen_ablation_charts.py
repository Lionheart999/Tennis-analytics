"""
Regenerate ablation bar charts from existing CSV results.

Usage:
    python scripts/regen_ablation_charts.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / 'runs' / 'eval'

CHARTS = [
    ('ablation_individual_results.csv', 'ablation_individual_chart.png', 'Individual Feature'),
    ('ablation_group_results.csv',      'ablation_group_chart.png',      'Feature Group'),
]


def draw_chart(df, chart_path, title_label):
    rows   = df[df['feature'] != 'baseline']
    labels = rows['feature'].tolist()
    drops  = rows['f1_drop'].tolist()
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in drops]

    baseline_f1 = df.loc[df['feature'] == 'baseline', 'val_f1'].iloc[0]

    fig_w = max(7, len(labels) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    bars = ax.bar(labels, drops, color=colors, width=0.6)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('F1 Drop (baseline − ablated)')
    ax.set_title(f'{title_label} Ablation  (baseline F1={baseline_f1:.3f})')
    ax.tick_params(axis='x', rotation=30)
    for bar, d in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                f'{d:+.3f}', ha='center', va='bottom', fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved {chart_path}")


for csv_name, chart_name, label in CHARTS:
    csv_path = OUT_DIR / csv_name
    if not csv_path.exists():
        print(f"SKIP — {csv_path} not found")
        continue
    df = pd.read_csv(csv_path)
    draw_chart(df, OUT_DIR / chart_name, label)
