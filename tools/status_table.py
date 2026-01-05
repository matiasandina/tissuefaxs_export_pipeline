#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rich.console import Console
from rich.table import Table


console = Console()

REGION_RE = re.compile(r"(Region)\s*(\d+)", re.IGNORECASE)
SUBJECT_RE = re.compile(r"^(sub-[A-Za-z0-9]+)")


@dataclass
class RegionRow:
    yyyymmdd: str
    subject_id: str
    project_folder: str
    relative_image_path: str
    exported: bool
    rotated: bool
    rgb: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan TissueFAXS exports for per-region status and write terminal/HTML/CSV tables."
    )
    p.add_argument(
        "root_dir",
        type=Path,
        help="Path to a yyyymmdd directory, or a parent directory when using --scan-root",
    )
    p.add_argument(
        "--scan-root",
        action="store_true",
        help="Treat root_dir as a parent containing yyyymmdd folders; run a report for each.",
    )
    p.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Output HTML report path (default: <root_dir>/tissuefaxs_status.html)",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV report path (default: <root_dir>/tissuefaxs_status.csv)",
    )
    return p.parse_args()


def find_project_roots(root_dir: Path) -> List[Path]:
    bfprojs = sorted(root_dir.rglob("*.bfproj"))
    roots = sorted({p.parent for p in bfprojs})
    return roots


def pick_root_dir(root_dir: Path, suffix: str) -> Optional[Path]:
    exact = root_dir / f"{root_dir.name}{suffix}"
    if exact.is_dir():
        return exact
    for p in sorted(root_dir.iterdir()):
        if p.is_dir() and p.name.endswith(suffix):
            return p
    return None


def subject_id_from_name(name: str) -> Optional[str]:
    m = SUBJECT_RE.match(name)
    if not m:
        return None
    return m.group(1)


def find_region_map(subject_dir: Path) -> Dict[str, Path]:
    inner = subject_dir / subject_dir.name
    candidates: List[Path] = []
    if inner.is_dir():
        candidates = sorted(inner.glob("Region *.tfcyto"))
    if not candidates:
        candidates = [p for p in subject_dir.rglob("*.tfcyto") if "Region" in p.name]

    region_map: Dict[str, Path] = {}
    for p in candidates:
        m = REGION_RE.search(p.name)
        if not m:
            continue
        num = int(m.group(2))
        label = f"Region {num:03d}"
        region_map.setdefault(label, p)
    return region_map


def regions_from_exports(
    subject_folder: str, exported_dir: Optional[Path], rotated_dir: Optional[Path]
) -> Dict[str, Path]:
    region_map: Dict[str, Path] = {}
    for base_dir in [exported_dir, rotated_dir]:
        if base_dir is None:
            continue
        for p in base_dir.glob(f"{subject_folder}_Region *_channels.tiff"):
            m = REGION_RE.search(p.name)
            if not m:
                continue
            num = int(m.group(2))
            label = f"Region {num:03d}"
            region_map.setdefault(label, p)
    return region_map


def expected_export_name(subject_folder: str, region_label: str) -> str:
    return f"{subject_folder}_{region_label}_channels.tiff"


def find_rgb_path(jpegs_dir: Optional[Path], subject_id: str, export_name: str) -> Optional[Path]:
    if jpegs_dir is None:
        return None
    base = export_name
    if base.lower().endswith(".tiff"):
        base = base[: -5]
    for ext in [".jpg", ".png", ".JPG", ".PNG"]:
        candidate = jpegs_dir / subject_id / f"{base}{ext}"
        if candidate.exists():
            return candidate
        candidate = jpegs_dir / f"{base}{ext}"
        if candidate.exists():
            return candidate
    return None


def render_table(rows: List[RegionRow]) -> None:
    table = Table(title="TissueFAXS export status")
    table.add_column("yyyymmdd", style="magenta")
    table.add_column("subject_id", style="cyan")
    table.add_column("project_folder")
    table.add_column("relative_image_path")
    table.add_column("exported", justify="center")
    table.add_column("rotated", justify="center")
    table.add_column("RGBs", justify="center")

    for row in rows:
        table.add_row(
            row.yyyymmdd,
            row.subject_id,
            row.project_folder,
            row.relative_image_path,
            "[green]yes[/green]" if row.exported else "[red]no[/red]",
            "[green]yes[/green]" if row.rotated else "[red]no[/red]",
            "[green]yes[/green]" if row.rgb else "[red]no[/red]",
        )

    console.print(table)


def print_open_html(path: Path) -> None:
    console.print(f"[bold]Open HTML:[/bold] file://{path}")


def write_csv(rows: List[RegionRow], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "yyyymmdd",
                "subject_id",
                "project_folder",
                "relative_image_path",
                "exported",
                "rotated",
                "RGBs",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.yyyymmdd,
                    row.subject_id,
                    row.project_folder,
                    row.relative_image_path,
                    "yes" if row.exported else "no",
                    "yes" if row.rotated else "no",
                    "yes" if row.rgb else "no",
                ]
            )


def write_html(rows: List[RegionRow], out_html: Path, root_dir: Path) -> None:
    total = len(rows)
    missing_export = sum(1 for r in rows if not r.exported)
    missing_rot = sum(1 for r in rows if not r.rotated)
    missing_rgb = sum(1 for r in rows if not r.rgb)
    last_ran = dt.datetime.now().isoformat(timespec="seconds")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TissueFAXS Status</title>
  <style>
    :root {{
      --bg: #f6f2e9;
      --ink: #1f2328;
      --muted: #5b646d;
      --ok: #1a7f37;
      --warn: #b35900;
      --bad: #b42318;
      --card: #fffdf9;
      --accent: #0b6ea8;
      --border: #dfd7c8;
    }}
    body {{
      margin: 0;
      padding: 32px;
      font-family: "IBM Plex Sans", "DejaVu Sans", "Helvetica Neue", Arial, sans-serif;
      background: radial-gradient(circle at 10% 10%, #ffffff 0%, var(--bg) 55%, #efe6d7 100%);
      color: var(--ink);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      letter-spacing: 0.3px;
    }}
    .subtitle {{
      margin: 0 0 18px;
      color: var(--muted);
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 12px;
      align-items: center;
      margin-bottom: 20px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .filters {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    label {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      font-size: 14px;
    }}
    input[type="text"] {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      font-size: 14px;
    }}
    .summary {{
      display: flex;
      gap: 16px;
      font-size: 14px;
      color: var(--muted);
    }}
    .hint {{
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    thead {{
      background: #f3eee4;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
    }}
    tbody tr:hover {{
      background: #f9f4ea;
    }}
    .pill {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.2px;
    }}
    .yes {{
      background: #e5f4eb;
      color: var(--ok);
    }}
    .no {{
      background: #fde9e7;
      color: var(--bad);
    }}
    .muted {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <h1>TissueFAXS Status</h1>
  <div class="subtitle">Root: {root_dir}</div>
  <div class="subtitle">Last ran: {last_ran}</div>
  <div class="controls">
    <div class="card">
      <input id="searchBox" type="text" placeholder="Filter by subject, project, region...">
    </div>
    <div class="card filters">
      <label><input type="checkbox" id="filterExport"> require exported</label>
      <label><input type="checkbox" id="filterRot"> require rotated</label>
      <label><input type="checkbox" id="filterRgb"> require RGBs</label>
      <div class="hint">Checked = only show rows with “yes”. Uncheck to include missing rows.</div>
    </div>
    <div class="card summary">
      <div>Total: {total}</div>
      <div>Missing export: {missing_export}</div>
      <div>Missing rotate: {missing_rot}</div>
      <div>Missing RGB: {missing_rgb}</div>
    </div>
  </div>

  <table id="statusTable">
    <thead>
      <tr>
        <th>yyyymmdd</th>
        <th>subject_id</th>
        <th>project_folder</th>
        <th>relative_image_path</th>
        <th>exported</th>
        <th>rotated</th>
        <th>RGBs</th>
      </tr>
    </thead>
    <tbody>
"""
    for row in rows:
        html += (
            "      <tr "
            f'data-exported="{str(row.exported).lower()}" '
            f'data-rotated="{str(row.rotated).lower()}" '
            f'data-rgb="{str(row.rgb).lower()}">\n'
        )
        html += f"        <td>{row.yyyymmdd}</td>\n"
        html += f"        <td>{row.subject_id}</td>\n"
        html += f"        <td>{row.project_folder}</td>\n"
        html += f"        <td class=\"muted\">{row.relative_image_path}</td>\n"
        html += f"        <td><span class=\"pill {'yes' if row.exported else 'no'}\">{'yes' if row.exported else 'no'}</span></td>\n"
        html += f"        <td><span class=\"pill {'yes' if row.rotated else 'no'}\">{'yes' if row.rotated else 'no'}</span></td>\n"
        html += f"        <td><span class=\"pill {'yes' if row.rgb else 'no'}\">{'yes' if row.rgb else 'no'}</span></td>\n"
        html += "      </tr>\n"

    html += """    </tbody>
  </table>

  <script>
    const searchBox = document.getElementById("searchBox");
    const filterExport = document.getElementById("filterExport");
    const filterRot = document.getElementById("filterRot");
    const filterRgb = document.getElementById("filterRgb");
    const rows = Array.from(document.querySelectorAll("#statusTable tbody tr"));

    function applyFilters() {
      const q = searchBox.value.trim().toLowerCase();
      const wantExport = filterExport.checked;
      const wantRot = filterRot.checked;
      const wantRgb = filterRgb.checked;

      rows.forEach((row) => {
        const text = row.textContent.toLowerCase();
        const matchesText = q === "" || text.includes(q);
        const okExport = !wantExport || row.dataset.exported === "true";
        const okRot = !wantRot || row.dataset.rotated === "true";
        const okRgb = !wantRgb || row.dataset.rgb === "true";
        row.style.display = matchesText && okExport && okRot && okRgb ? "" : "none";
      });
    }

    [searchBox, filterExport, filterRot, filterRgb].forEach((el) => {
      el.addEventListener("input", applyFilters);
      el.addEventListener("change", applyFilters);
    });
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def build_rows(
    root_dir: Path,
    project_roots: Iterable[Path],
    exported_dir: Optional[Path],
    rotated_dir: Optional[Path],
    jpegs_dir: Optional[Path],
    day_label: str,
) -> List[RegionRow]:
    rows: List[RegionRow] = []
    for project_root in project_roots:
        subject_dirs = [
            d
            for d in sorted(project_root.iterdir())
            if d.is_dir() and d.name.startswith("sub-")
        ]
        for subject_dir in subject_dirs:
            subject_folder = subject_dir.name
            subject_id = subject_id_from_name(subject_folder) or subject_folder
            region_map = find_region_map(subject_dir)
            if not region_map:
                region_map = regions_from_exports(
                    subject_folder, exported_dir, rotated_dir
                )
            for region_label, region_path in sorted(region_map.items()):
                export_name = expected_export_name(subject_folder, region_label)
                export_path = exported_dir / export_name if exported_dir else None
                rot_path = rotated_dir / export_name if rotated_dir else None
                rgb_path = find_rgb_path(jpegs_dir, subject_id, export_name)

                try:
                    rel_path = region_path.relative_to(project_root)
                    rel_display = str(rel_path)
                except ValueError:
                    rel_display = export_name

                rows.append(
                    RegionRow(
                        yyyymmdd=day_label,
                        subject_id=subject_id,
                        project_folder=project_root.name,
                        relative_image_path=rel_display,
                        exported=bool(export_path and export_path.exists()),
                        rotated=bool(rot_path and rot_path.exists()),
                        rgb=bool(rgb_path and rgb_path.exists()),
                    )
                )
    return rows


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    if not root_dir.is_dir():
        raise SystemExit(f"Root directory does not exist: {root_dir}")

    if args.scan_root:
        yyyymmdd_dirs = sorted(
            [d for d in root_dir.iterdir() if d.is_dir() and re.fullmatch(r"\d{8}", d.name)]
        )
        if not yyyymmdd_dirs:
            raise SystemExit(f"No yyyymmdd directories found under {root_dir}")

        all_rows: List[RegionRow] = []
        total = len(yyyymmdd_dirs)
        for i, day_dir in enumerate(yyyymmdd_dirs, start=1):
            console.rule(f"[bold]Folder {i}/{total} · {day_dir.name}[/bold]")
            try:
                day_rows = collect_rows(day_dir, day_dir.name)
            except SystemExit as e:
                console.print(f"[red]Skipped {day_dir.name}:[/red] {e}")
                continue
            all_rows.extend(day_rows)

        if not all_rows:
            raise SystemExit("No regions found in any yyyymmdd directories.")

        render_table(all_rows)
        out_html = args.out_html or (root_dir / "tissuefaxs_status.html")
        out_csv = args.out_csv or (root_dir / "tissuefaxs_status.csv")
        write_html(all_rows, out_html, root_dir)
        write_csv(all_rows, out_csv)
        console.print(f"[bold]HTML written:[/bold] {out_html}")
        console.print(f"[bold]CSV written:[/bold] {out_csv}")
        print_open_html(out_html)
        return

    run_single(root_dir, None, args.out_html, args.out_csv)


def run_single(
    root_dir: Path,
    report_dir: Optional[Path],
    out_html_arg: Optional[Path],
    out_csv_arg: Optional[Path],
) -> None:
    rows = collect_rows(root_dir, root_dir.name)
    render_table(rows)
    if report_dir is not None:
        out_html = report_dir / f"{root_dir.name}_tissuefaxs_status.html"
        out_csv = report_dir / f"{root_dir.name}_tissuefaxs_status.csv"
    else:
        out_html = out_html_arg or (root_dir / "tissuefaxs_status.html")
        out_csv = out_csv_arg or (root_dir / "tissuefaxs_status.csv")
    write_html(rows, out_html, root_dir)
    write_csv(rows, out_csv)

    console.print(f"[bold]HTML written:[/bold] {out_html}")
    console.print(f"[bold]CSV written:[/bold] {out_csv}")
    print_open_html(out_html)

    missing_export = sum(1 for r in rows if not r.exported)
    missing_rot = sum(1 for r in rows if not r.rotated)
    missing_rgb = sum(1 for r in rows if not r.rgb)

    exported_dir = pick_root_dir(root_dir, "_exported_tif")
    rotated_dir = pick_root_dir(root_dir, "_rotated_tif")
    if missing_export:
        console.print(
            "[yellow]Missing exports detected. Export from TissueFAXS into the exported_tif folder.[/yellow]"
        )
    if missing_rot and exported_dir and rotated_dir:
        console.print(
            "[yellow]Rotation suggestion:[/yellow]\n"
            f"  .venv/bin/python 01-rotate_all_regions.py --input_dir \"{exported_dir}\" "
            f"--output_dir \"{rotated_dir}\""
        )
    if missing_rgb and rotated_dir:
        console.print(
            "[yellow]RGB composites suggestion:[/yellow]\n"
            f"  .venv/bin/python 02-make_overviews.py \"{rotated_dir}\" --format jpg --split-view"
        )


def collect_rows(root_dir: Path, day_label: str) -> List[RegionRow]:
    project_roots = find_project_roots(root_dir)
    if not project_roots:
        raise SystemExit(f"No .bfproj files found under {root_dir}")

    exported_dir = pick_root_dir(root_dir, "_exported_tif")
    rotated_dir = pick_root_dir(root_dir, "_rotated_tif")
    jpegs_dir = root_dir / "JPEGs" if (root_dir / "JPEGs").is_dir() else None

    rows = build_rows(
        root_dir, project_roots, exported_dir, rotated_dir, jpegs_dir, day_label
    )
    if not rows:
        raise SystemExit("No regions found. Check project structure or exports.")
    return rows

 
if __name__ == "__main__":
    main()
