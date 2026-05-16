#!/bin/bash
#SBATCH --job-name=csi_cnn
#SBATCH --output=logs/csi_cnn_%j.out
#SBATCH --error=logs/csi_cnn_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# ── Setup ─────────────────────────────────────────────────────────────────────
set -e
echo "=== CSI CNN Training Job $SLURM_JOB_ID ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"

# Activate environment (adjust path to your cluster venv)
source $HOME/venvs/tf/bin/activate

# Go to project root (scp/rsync your repo here before submitting)
cd $HOME/defence_2026

# Install deps if needed (first run only)
pip install -q -r ml/requirements-train.txt

# ── Analyse data first ────────────────────────────────────────────────────────
echo ""
echo "--- Data analysis ---"
python -m ml.csi.analyze --npz data/csi_windows.npz

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "--- Training ---"
python -m ml.csi.train \
    --data   data/csi_windows.npz \
    --out    models/csi_cnn \
    --epochs 40 \
    --batch  32 \
    --qat

# ── Export TFLite ─────────────────────────────────────────────────────────────
echo ""
echo "--- TFLite export ---"
python -m ml.csi.export_tflite --model-dir models/csi_cnn

echo ""
echo "=== Done. Outputs in models/csi_cnn/ ==="
