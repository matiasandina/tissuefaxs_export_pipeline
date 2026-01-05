# process_tissue_faxs

## Why this exists
TissueFAXS exports OME-TIFF regions, but the exported regions are often rotated 180°. This repo provides a small pipeline to:

1) validate exports
2) rotate multi‑channel tiles, and 
3) create overview composites and montages for quick visual QC.

## Pipeline overview
- `00-validator.py`: Validate per-region TIFF exports for consistent shapes and physical pixel size.
- `01-rotate_all_regions.py`: Rotate multi‑channel `*_channels.tiff` tiles by 180° and write pyramidal TIFFs.
- `02-make_overviews.py`: Create per‑tile RGB composites, optional split‑channel views, and per‑subject montages.

## Repo layout
- `00-validator.py`, `01-rotate_all_regions.py`, `02-make_overviews.py`: entrypoints.
- `src/`: shared helpers (TIFF IO, pyramids, metadata).
- `jobs/`: SLURM batch wrappers.
- `tools/`: one-off inspection helpers (including status reporting).

## Setup
- Create the venv in repo root (required):
  - `uv venv --python 3.11.6 .venv`
- Install dependencies:
  - `uv pip install -r requirements.txt`

## Inputs, outputs, assumptions
### Inputs
- Exports should be OME‑TIFFs. Rotation and overview steps expect `*_channels.tiff` files with channels stacked as `(C, Y, X)`.
- Subject IDs are inferred from filenames via `^sub-[A-Za-z0-9]+_`.
- Region labels are inferred via `Region <num>` in the filename.

### Outputs
- Rotated tiles are written to the directory passed with `--output_dir` (or in‑place if omitted).
- Overview composites are written under the `--out-jpegs` directory, in a subfolder per subject ID.
- Split‑channel views (if enabled) are written alongside composites as `<basename>_split.<fmt>`.
- Montages are written to the `--out-montages` directory as `Montage_<subject>.<fmt>`.
**Warning:** rotation currently writes QuPath‑style pyramidal TIFFs without OME metadata, so physical pixel sizes and other OME metadata are not preserved.

## Typical workflow
1) Validate exports:
   - `.venv/bin/python 00-validator.py /path/to/exports`
2) Rotate tiles:
   - `.venv/bin/python 01-rotate_all_regions.py --input_dir /path/to/exports --output_dir /path/to/rotated`
3) Create overviews:
   - `.venv/bin/python 02-make_overviews.py /path/to/rotated --config config.yaml`

## Run locally
- Validate:
  - `.venv/bin/python 00-validator.py /path/to/input_dir`
- Rotate:
  - `.venv/bin/python 01-rotate_all_regions.py --input_dir /path/to/input_dir --output_dir /path/to/out`
- Make overviews:
  - `.venv/bin/python 02-make_overviews.py /path/to/rotated --out-jpegs /path/to/JPEGs --out-montages /path/to/montages --split-view --label basename`
- Status report (single day folder):
  - `.venv/bin/python tools/status_table.py /path/to/yyyymmdd`
- Status report (parent folder containing yyyymmdd):
  - `.venv/bin/python tools/status_table.py /path/to/parent --scan-root`
  - Outputs are written as `tissuefaxs_status.html` and `tissuefaxs_status.csv` in the target directory.

### Config file for 02 (recommended)
You can use a YAML config to keep defaults in one place. CLI flags always override the config.

1) Copy the template:
   - `cp config_template.yaml config.yaml`
2) Edit `config.yaml` as needed.
3) Run:
   - `.venv/bin/python 02-make_overviews.py /path/to/rotated --config config.yaml`

Example for DAPI+signal (blue + red, no green):
```yaml
ch_blue: 0
ch_red: 1
ch_green: -1
split_view: true
label: basename
```

### Debug single tile (02)
- Use `--only` with a filename glob:
  - `.venv/bin/python 02-make_overviews.py /path/to/rotated --only "*Region 001*" --out-jpegs /path/to/JPEGs --out-montages /path/to/montages`

## Run with SLURM (sbatch)
Batch jobs are in `jobs/`. They forward all arguments after `--` to the Python script.

Examples:
- Validate:
  - `sbatch jobs/00-validator.sbatch /path/to/input_dir`
- Rotate:
  - `sbatch jobs/01-rotate_all_regions.sbatch --input_dir /path/to/input_dir --output_dir /path/to/out`
- Make overviews:
  - `sbatch jobs/02-make_overviews.sbatch /path/to/rotated --out-jpegs /path/to/JPEGs --out-montages /path/to/montages --split-view --label basename`
- Make overviews (with config):
  - `sbatch jobs/02-make_overviews.sbatch /path/to/rotated --config config.yaml`

### Venv location for batch jobs
The batch wrappers look for `.venv/` in the submit directory. If your venv is elsewhere, pass `VENV_PATH`:
- `VENV_PATH=/path/to/.venv sbatch jobs/00-validator.sbatch -- /path/to/input_dir`

## Key options for 02
- `--config <path>`: load YAML config; CLI flags override config values.
- `--split-view`: save side‑by‑side channel 0/1 views next to composites.
- `--label {basename,region,none}`: choose label text burned onto tiles.
- `--only "<glob>"`: process a single tile or subset for debugging.
- `--tile-max-side`: downscale large tiles before saving.

## Script reference
### 00-validator.py
Validates each region for:
- consistent spatial dimensions across channel files
- consistent physical pixel sizes (OME metadata)
- presence of at least one non-RGB scientific channel

### 01-rotate_all_regions.py
Rotates each `*_channels.tiff` by 180 degrees and writes pyramidal TIFFs.
Key options:
- `--input_dir`: directory with `*_channels.tiff` tiles (required)
- `--output_dir`: output directory (if omitted, overwrites in place)
- `--dry-run`: print actions only
**Warning:** output uses QuPath‑style pyramidal TIFFs without OME metadata, so OME physical sizes are not preserved.

### 02-make_overviews.py
Creates RGB composites and montages for QC.
Key options:
- `--config`: YAML file with defaults
- `--out-jpegs`, `--out-montages`: output directories
- `--ch-blue`, `--ch-red`, `--ch-green`: channel mapping to RGB
- `--split-view`: save channel 0/1 split view

## Known limitations
- The subject ID regex is strict. If your filenames differ, update `SUBJECT_RE` in `02-make_overviews.py`.
- Label `basename` can be long; use `--label region` if you prefer shorter labels.
