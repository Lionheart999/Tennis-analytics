#!/bin/bash
# Stages all source code, results, and configs to git,
# and updates DVC tracking for large binary files.
set -e
cd "$(dirname "$0")"

echo "=== Staging source code ==="
git add scripts/ablation.py
git add scripts/build_dataset_cnn.py
git add scripts/eval_model_cnn.py
git add scripts/extract_cnn_embeddings.py
git add scripts/train_gru_cnn.py
git add scripts/eval_held_out.py
git add scripts/eval_segments_held_out.py
git add scripts/regen_ablation_charts.py
git add scripts/sweep_buffer.py
git add scripts/time_pipelines.py
git add brev_setup.sh

echo "=== Staging labels ==="
git add labels/

echo "=== Staging model configs and training history ==="
git add models/rally_gru_config.txt
git add models/rally_gru_config_v1.txt
git add models/rally_gru_cnn_config.txt
git add models/training_history.csv
git add models/training_history_cnn.csv

echo "=== Staging eval results and run outputs ==="
git add runs/eval/
git add runs/results.csv
git add runs/timing_results.txt

echo "=== Staging new model weights ==="
# models/ already has files tracked by git, so keep .pt files in git too (56M total)
git add models/

echo "=== Updating DVC tracking for large binary files ==="

# Track embeddings (6.2G)
dvc add embeddings/
git add embeddings.dvc

# features/ already has files tracked by git — just add the new CSVs directly
git add features/

# Track CNN dataset (8M)
dvc add gru_dataset_cnn/
git add gru_dataset_cnn.dvc

echo "=== Committing to git ==="
git commit -m "Add CNN pipeline, eval results, model configs, and updated DVC pointers"

echo ""
echo "Done. To push large files to DVC remote storage, run:"
echo "  dvc push"
