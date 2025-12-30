#!/bin/bash
#SBATCH --job-name=txf-01-rotate
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=24

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

VENV="${VENV_PATH:-$PWD/.venv}"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv at $VENV (set VENV_PATH to override)." >&2
  exit 1
fi

"$PY" 01-rotate_all_regions.py "$@"
