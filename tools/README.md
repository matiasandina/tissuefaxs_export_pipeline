# Tools

## status_table.py
Generate an at-a-glance status table that checks whether exports, rotated TIFFs,
and RGB composites exist for each Region in a TissueFAXS project. The HTML output
includes filters and a timestamp.

### Usage
Single day folder:
- `.venv/bin/python tools/status_table.py /path/to/yyyymmdd`

Parent folder with multiple `yyyymmdd` subfolders:
- `.venv/bin/python tools/status_table.py /path/to/parent --scan-root`

Outputs are written as `tissuefaxs_status.html` and `tissuefaxs_status.csv` in the
target directory.

### Example (sanitized, downsampled)
```
$ .venv/bin/python tools/status_table.py /path/to/histology --scan-root
──────────────────────────── Folder 1/3 · 20250721 ────────────────────────────
──────────────────────────── Folder 2/3 · 20251126 ────────────────────────────
Skipped 20251126: No regions found. Check project structure or exports.
──────────────────────────── Folder 3/3 · 20251208 ────────────────────────────

TissueFAXS export status
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ yyyymmdd ┃ subject_id ┃ project_folder   ┃ relative_image_path                 ┃ exported ┃ rotated ┃ RGBs ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ 20250721 │ sub-MLA357 │ 2025-07-21 12-50 │ sub-MLA357/.../Region 001.tfcyto     │    no    │   no    │  no  │
│ 20250721 │ sub-MLA358 │ 2025-07-21 12-50 │ sub-MLA358/.../Region 002.tfcyto     │    no    │   no    │  no  │
│ 20250721 │ sub-MLA361 │ 2025-07-21 12-50 │ sub-MLA361/.../Region 003.tfcyto     │    no    │   no    │  no  │
│ 20251208 │ sub-MLA437 │ 2025-12-08 11-42 │ sub-MLA437/.../Region 001.tfcyto     │   yes    │   yes   │ yes  │
│ 20251208 │ sub-MLA438 │ 2025-12-08 11-42 │ sub-MLA438/.../Region 004.tfcyto     │   yes    │   yes   │ yes  │
│ 20251208 │ sub-MLA496 │ 2025-12-23 09-40 │ sub-MLA496/.../Region 004.tfcyto     │   yes    │   yes   │  no  │
└──────────┴────────────┴──────────────────┴──────────────────────────────────────┴──────────┴─────────┴──────┘

HTML written: /path/to/histology/tissuefaxs_status.html
CSV written: /path/to/histology/tissuefaxs_status.csv
Open HTML: file:///path/to/histology/tissuefaxs_status.html
```
