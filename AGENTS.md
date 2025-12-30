# Process TissueFAXS

## Project layout
- Main entrypoints live at repo root: `00-validator.py`, `01-rotate_all_regions.py`, `02-make_overviews.py`.
- Shared utilities live in `src/`.
- SLURM batch jobs live in `jobs/`.
- Dev/ad-hoc helpers live in `tools/`.

## Environment
- Preferred venv location: `.venv/` at repo root.
- `requirements.txt` is provided for conda or other environments.

## Running
- Local: use `.venv/bin/python 00-validator.py ...` etc.
- Batch: `sbatch jobs/00-validator.sbatch -- <args>` (args are forwarded).
