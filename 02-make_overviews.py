#!/usr/bin/env python3
"""
2-channel *_channels.tiff -> per-ID normalized JPEG/PNG composites -> per-ID montages.

Assumptions:
- Inputs are already rotated (or you don't care about rotation here).
- Each file contains 2 channels, and we read T=0, Z=0.
- Channel mapping default: ch0 -> Green, ch1 -> Red (configurable).
- Sample ID is the prefix before the first underscore in the filename.

Install:
  pip install bioio bioio-ome-tiff pillow rich numpy

Example:
  python 02-make_overviews.py /abs/path/to/rotated \
    --pattern "*_channels.tif*" \
    --out-jpegs /abs/path/to/JPEGs \
    --out-montages /abs/path/to/Montages \
    --format jpg \
    --split-view \
    --label basename
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TextColumn,
)
import tifffile as tf

from rich.table import Table

from bioio import BioImage


# display presets (tuned for DAPI + sparse signal)
DAPI_P_LOW  = 0.00
DAPI_P_HIGH = 99.99
DAPI_GAMMA  = 1.0
DAPI_HARD_MAX = 60000

SIG_P_LOW   = 0
SIG_P_HIGH  = 99.9
SIG_GAMMA   = 1
# Use this to bypass estimation and get the point where signal saturates
SIG_WHITE_POINT = 15000.0

console = Console()


DEFAULTS = {
    "pattern": "*_channels.tif*",
    "out_jpegs": None,
    "out_montages": None,
    "format": "jpg",
    "overwrite": False,
    "split_view": False,
    "label": "basename",
    "only": None,
    "per_id_agg": "median",
    "tile_max_side": 2048,
    "ch_blue": 0,
    "ch_green": -1,
    "ch_red": 1,
    "max_montage_pixels": 1_600_000_000,
    "max_montage_side": 30_000,
}


@dataclass
class IdStats:
    max_w: int = 0
    max_h: int = 0
    # per-channel robust min/max (display scaling)
    lo: Optional[np.ndarray] = None  # shape (2,)
    hi: Optional[np.ndarray] = None  # shape (2,)


def load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "Missing PyYAML; install it or remove --config. "
            "Example: uv pip install pyyaml"
        ) from e

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise SystemExit("Config file must be a mapping at the top level.")

    # Normalize keys to match argparse dests.
    normalized: Dict[str, Any] = {}
    for k, v in data.items():
        normalized[k.replace("-", "_")] = v
    return normalized


def merge_config(defaults: Dict[str, Any], cfg: Dict[str, Any], cli: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(cfg)
    for k, v in cli.items():
        if v is not None:
            merged[k] = v
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create overview JPEG/PNG composites and per-ID montages from 2-channel *_channels.tiff tiles."
    )
    p.add_argument("input_dir", type=Path, help="Directory containing rotated *_channels.tiff tiles.")
    p.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    p.add_argument("--pattern", default=None, help="Glob pattern to match input files.")
    p.add_argument(
        "--only",
        default=None,
        help="Optional filename glob to process a single tile (e.g. '*Region 001*').",
    )
    p.add_argument("--out-jpegs", type=Path, default=None, help="Output directory for per-tile JPEG/PNG composites.")
    p.add_argument("--out-montages", type=Path, default=None, help="Output directory for per-ID montages.")
    p.add_argument("--format", choices=["jpg", "png"], default=None, help="Output image format for composites/montages.")
    p.add_argument("--overwrite", action="store_true", default=None, help="Overwrite existing outputs.")
    p.add_argument(
        "--split-view",
        action="store_true",
        default=None,
        help="Also save a side-by-side split of channel 0 and 1 alongside the composite.",
    )
    p.add_argument(
        "--label",
        choices=["region", "basename", "none"],
        default=None,
        help="Label to burn onto tiles (default: basename).",
    )
    #p.add_argument("--id-sep", default="_", help="Separator to define ID prefix (default: underscore).")

    # display scaling
    p.add_argument(
        "--per-id-agg",
        choices=["median", "mean"],
        default=None,
        help="Aggregate per-tile percentiles into per-ID scaling using median or mean.",
    )
    p.add_argument(
        "--tile-max-side",
        type=int,
        default=None,
        help="Downscale tiles so max(width, height) <= this before saving (e.g. 2048).",
    )
    # channel mapping to RGB
    p.add_argument("--ch-green", type=int, default=None, help="Channel index to map to Green (default -1).")
    p.add_argument("--ch-red", type=int, default=None, help="Channel index to map to Red (default 1).")
    p.add_argument("--ch-blue", type=int, default=None, help="Channel index to map to Blue (-1 to leave blue=0).")

    # montage limits
    p.add_argument(
        "--max-montage-pixels",
        type=int,
        default=None,
        help="Approx pixel budget to avoid massive allocations (default ~1.6e9).",
    )
    p.add_argument(
        "--max-montage-side",
        type=int,
        default=None,
        help="If montage width or height exceeds this, downscale tiles (default 30000).",
    )

    args = p.parse_args()

    cfg = {}
    if args.config is not None:
        cfg = load_yaml_config(args.config)

    cli = vars(args).copy()
    cli.pop("config", None)
    merged = merge_config(DEFAULTS, cfg, cli)

    # Rebuild a namespace with merged values
    args = argparse.Namespace(**merged)
    return args


def list_inputs(input_dir: Path, pattern: str, only: Optional[str]) -> List[Path]:
    files = sorted(input_dir.glob(pattern))
    # filter common junk
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]
    if only:
        files = [f for f in files if fnmatch.fnmatch(f.name, only)]
    return files


import re
REGION_RE = re.compile(r"(Region\s*\d+)", re.IGNORECASE)

def extract_region_label(name: str) -> Optional[str]:
    m = REGION_RE.search(name)
    return m.group(1) if m else None

SUBJECT_RE = re.compile(r"^(sub-[A-Za-z0-9]+)_")

def get_id_from_name(name: str) -> Optional[str]:
    m = SUBJECT_RE.match(name)
    if not m:
        return None
    return m.group(1)


def label_for_file(path: Path, mode: str) -> Optional[str]:
    if mode == "none":
        return None
    if mode == "region":
        return extract_region_label(path.name)
    if mode == "basename":
        return path.stem
    raise ValueError(f"Unknown label mode: {mode}")


def burn_label(
    rgb: np.ndarray,
    text: str,
    *,
    margin: int = 10,
    font_size: int = 48,
    fill=(255, 255, 255),
    bg=(0, 0, 0),
    bg_alpha: int = 160,
) -> np.ndarray:
    im = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(im)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]

    x = margin
    y = margin

    # background rectangle
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle(
        [x - 6, y - 4, x + text_w + 6, y + text_h + 4],
        fill=(*bg, bg_alpha),
    )

    im = Image.alpha_composite(im, overlay)
    draw = ImageDraw.Draw(im)
    draw.text((x, y), text, fill=fill + (255,), font=font)

    return np.array(im.convert("RGB"))

def safe_read_cyx(path: Path) -> np.ndarray:
    """
    Returns image as (C, Y, X).
    Tries BioIO first (for real OME-TIFFs),
    falls back to tifffile for broken/derived TIFFs.
    """
    try:
        img = BioImage(str(path))
        arr = img.get_image_data("CZYX", T=0)
        return arr[:, 0, :, :]
    except Exception as e:
        # fallback: raw TIFF
        with tf.TiffFile(path) as tif:
            data = tif.asarray()

        # Normalize shape
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 3:
            # assume (Y, X, C) or (C, Y, X)
            if data.shape[0] in (1, 2, 3, 4):
                pass  # already C,Y,X
            else:
                data = np.moveaxis(data, -1, 0)
        else:
            raise ValueError(f"Unsupported TIFF shape {data.shape} for {path}")

        return data

def compute_tile_percentiles(cyx: np.ndarray, p_low: float, p_high: float) -> Tuple[np.ndarray, np.ndarray]:
    if cyx.shape[0] < 2:
        raise ValueError(f"Expected >=2 channels, got {cyx.shape[0]}")
    lo = np.percentile(cyx[:2].reshape(2, -1), p_low, axis=1)
    hi = np.percentile(cyx[:2].reshape(2, -1), p_high, axis=1)
    # guard: avoid degenerate scaling
    hi = np.maximum(hi, lo + 1e-6)
    return lo.astype(np.float64), hi.astype(np.float64)


def pad_center_cyx(cyx: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    c, h, w = cyx.shape
    if w == target_w and h == target_h:
        return cyx
    if w > target_w or h > target_h:
        return cyx

    out = np.zeros((c, target_h, target_w), dtype=cyx.dtype)
    y0 = (target_h - h) // 2
    x0 = (target_w - w) // 2
    out[:, y0 : y0 + h, x0 : x0 + w] = cyx
    return out


def estimate_display_limits(
    channel: np.ndarray,
    p_low: float,
    p_high: float,
):
    x = channel[channel > 0]
    if x.size < 100:
        return 0.0, float(channel.max())

    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)

    if hi <= lo:
        hi = lo + 1.0

    return float(lo), float(hi)

def apply_gamma(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    return np.power(x, gamma)

def scale_channel(chan, lo, hi, gamma=1.0):
    x = np.clip(chan.astype(np.float32), lo, hi)
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0, 1)
    x = np.power(x, gamma)
    return (255 * x).astype(np.uint8)


def scale_to_u8(chan: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.clip(chan.astype(np.float32), lo, hi)
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def make_rgb(cyx: np.ndarray, lo: np.ndarray, hi: np.ndarray, ch_r: int, ch_g: int, ch_b: int) -> np.ndarray:
    c, y, x = cyx.shape
    rgb = np.zeros((y, x, 3), dtype=np.uint8)

    def get_ch(idx: int) -> Optional[np.ndarray]:
        if idx is None or idx < 0:
            return None
        if idx >= c:
            raise ValueError(f"Requested channel {idx} but only {c} channels present")
        return cyx[idx]

    # scaling is computed for channels 0 and 1; if user maps differently, fall back to per-channel min/max
    def get_scale(idx: int) -> Tuple[float, float]:
        if idx in (0, 1):
            return float(lo[idx]), float(hi[idx])
        ch = cyx[idx].astype(np.float32)
        a = float(np.percentile(ch, 1.0))
        b = float(np.percentile(ch, 99.8))
        if b <= a:
            b = a + 1e-6
        return a, b

    for out_idx, ch_idx in [(0, ch_r), (1, ch_g), (2, ch_b)]:
        if ch_idx is None or ch_idx < 0:
            continue
        a, b = get_scale(ch_idx)
        gamma = (
            DAPI_GAMMA if ch_idx == 0
            else SIG_GAMMA
        )
        rgb[:, :, out_idx] = scale_channel(cyx[ch_idx], a, b, gamma=gamma)

    return rgb


def downscale_if_needed(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    new_w = int(w * scale)
    new_h = int(h * scale)
    return np.array(
        Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
    )


def save_image(rgb: np.ndarray, out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(rgb, mode="RGB")
    if fmt == "jpg":
        im.save(out_path, quality=92, subsampling=0, optimize=True)
    else:
        im.save(out_path, compress_level=6)


def make_split_view(cyx: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Side-by-side split view of channel 0 and 1, scaled to 8-bit grayscale.
    """
    if cyx.shape[0] < 2:
        raise ValueError("Split view requires at least 2 channels.")

    ch0 = scale_to_u8(cyx[0], float(lo[0]), float(hi[0]))
    ch1 = scale_to_u8(cyx[1], float(lo[1]), float(hi[1]))

    g0 = np.repeat(ch0[:, :, None], 3, axis=2)
    g1 = np.repeat(ch1[:, :, None], 3, axis=2)
    return np.concatenate([g0, g1], axis=1)


def grid_shape(n: int) -> Tuple[int, int]:
    cols = int(math.floor(math.sqrt(n)))
    if cols * cols < n:
        cols += 1
    rows = int(math.floor((n + cols - 1) / cols))
    return cols, rows


def build_montage(images: List[Path], out_path: Path, max_side: int, max_pixels: int, fmt: str) -> None:
    # assumes all tiles have identical dims
    ims = [Image.open(p).convert("RGB") for p in images]
    if not ims:
        return
    w, h = ims[0].size
    n = len(ims)
    cols, rows = grid_shape(n)

    montage_w = cols * w
    montage_h = rows * h

    # downscale if huge (either by side constraint or rough pixel budget)
    scale = 1.0
    if montage_w > max_side or montage_h > max_side:
        scale = min(max_side / montage_w, max_side / montage_h)

    pix_est = float(montage_w) * float(montage_h)
    if pix_est > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / pix_est))

    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        ims = [im.resize((new_w, new_h), resample=Image.BILINEAR) for im in ims]
        w, h = new_w, new_h
        montage_w = cols * w
        montage_h = rows * h

    canvas = Image.new("RGB", (montage_w, montage_h), (0, 0, 0))
    for i, im in enumerate(ims):
        r = i // cols
        c = i % cols
        canvas.paste(im, (c * w, r * h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jpg":
        canvas.save(out_path, quality=92, subsampling=0, optimize=True)
    else:
        canvas.save(out_path, compress_level=6)

    for im in ims:
        im.close()


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input dir does not exist: {input_dir}")

    out_jpegs = args.out_jpegs or (input_dir.parent / "JPEGs")
    out_montages = args.out_montages or (input_dir.parent / "montages")
    fmt = args.format

    files = list_inputs(input_dir, args.pattern, args.only)
    if not files:
        only_msg = f" and only='{args.only}'" if args.only else ""
        raise SystemExit(f"No files matched pattern='{args.pattern}' in {input_dir}{only_msg}")

    # group files by ID
    by_id: Dict[str, List[Path]] = {}
    for f in files:
        sid = get_id_from_name(f.name)
        if sid is None:
            continue
        by_id.setdefault(sid, []).append(f)

    if not by_id:
        raise SystemExit(f"No files contained the subject ID regex to infer IDs.")

    # Pass 1: per-ID max canvas and per-tile percentiles (then aggregate per ID)
    stats: Dict[str, IdStats] = {sid: IdStats() for sid in by_id.keys()}
    lo_acc: Dict[str, List[np.ndarray]] = {sid: [] for sid in by_id.keys()}
    hi_acc: Dict[str, List[np.ndarray]] = {sid: [] for sid in by_id.keys()}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        t1 = progress.add_task("Scan tiles (dims + robust percentiles)", total=len(files))
        for f in files:
            sid = get_id_from_name(f.name)
            if sid is None or sid not in stats:
                progress.advance(t1)
                continue

            cyx = safe_read_cyx(f)
            _, y, x = cyx.shape
            st = stats[sid]
            st.max_w = max(st.max_w, x)
            st.max_h = max(st.max_h, y)
            
            # assumes counterstain is channel 0, signal channel is 1!!
            lo0, hi0 = estimate_display_limits(cyx[0], DAPI_P_LOW, DAPI_P_HIGH)
            hi0 = min(hi0, DAPI_HARD_MAX)
            lo1, hi1 = estimate_display_limits(cyx[1], SIG_P_LOW, SIG_P_HIGH)
            hi1 = max(hi1, SIG_WHITE_POINT)

            lo = np.array([lo0, lo1])
            hi = np.array([hi0, hi1])
            lo_acc[sid].append(lo)
            hi_acc[sid].append(hi)

            progress.advance(t1)

    for sid in stats.keys():
        lo_stack = np.stack(lo_acc[sid], axis=0)  # (n_tiles, 2)
        hi_stack = np.stack(hi_acc[sid], axis=0)
        if args.per_id_agg == "median":
            stats[sid].lo = np.median(lo_stack, axis=0)
            stats[sid].hi = np.median(hi_stack, axis=0)
        else:
            stats[sid].lo = np.mean(lo_stack, axis=0)
            stats[sid].hi = np.mean(hi_stack, axis=0)

        # guard against degeneracy
        stats[sid].hi = np.maximum(stats[sid].hi, stats[sid].lo + 1e-6)

    # show summary
    tbl = Table(title="Per-ID normalization targets")
    tbl.add_column("ID")
    tbl.add_column("Canvas (W×H)", justify="right")
    tbl.add_column("ch0 lo-hi", justify="right")
    tbl.add_column("ch1 lo-hi", justify="right")
    for sid, st in sorted(stats.items()):
        tbl.add_row(
            sid,
            f"{st.max_w}×{st.max_h}",
            f"{st.lo[0]:.2f}-{st.hi[0]:.2f}",
            f"{st.lo[1]:.2f}-{st.hi[1]:.2f}",
        )
    console.print(tbl)

    # Pass 2: export composites
    exported: List[Path] = []
    with progress:
        t2 = progress.add_task("Export composites", total=len(files))
        for f in files:
            sid = get_id_from_name(f.name)
            if sid is None or sid not in stats:
                progress.advance(t2)
                continue

            out_name = f.name
            # strip suffixes like .tif/.tiff, keep base
            base = out_name
            for ext in [".tiff", ".tif", ".TIFF", ".TIF"]:
                if base.endswith(ext):
                    base = base[: -len(ext)]
                    break
            sid_dir = out_jpegs / sid
            out_path = sid_dir / f"{base}.{fmt}"
            split_path = sid_dir / f"{base}_split.{fmt}"

            if out_path.exists() and not args.overwrite:
                console.print(f"[yellow]Skip existing:[/yellow] {out_path}")
                progress.advance(t2)
                continue

            cyx = safe_read_cyx(f)
            st = stats[sid]
            cyx = pad_center_cyx(cyx, st.max_w, st.max_h)
            rgb = make_rgb(
                cyx,
                st.lo,
                st.hi,
                ch_r=args.ch_red,
                ch_g=args.ch_green,
                ch_b=args.ch_blue,
            )
            # burn label
            label = label_for_file(f, args.label)
            if label:
                rgb = burn_label(rgb, label)


            # downsize
            if args.tile_max_side is not None:
                rgb = downscale_if_needed(rgb, args.tile_max_side)
            save_image(rgb, out_path, fmt)
            exported.append(out_path)

            if args.split_view:
                split_rgb = make_split_view(cyx, st.lo, st.hi)
                if label:
                    split_rgb = burn_label(split_rgb, label)
                if args.tile_max_side is not None:
                    split_rgb = downscale_if_needed(split_rgb, args.tile_max_side)
                save_image(split_rgb, split_path, fmt)

            progress.advance(t2)

    # Pass 3: montages per ID
    with progress:
        t3 = progress.add_task("Build montages", total=len(by_id))
        for sid, flist in sorted(by_id.items()):
            # map tile paths to exported composite paths
            comp_paths: List[Path] = []
            sid_dir = out_jpegs / sid
            for f in sorted(flist):
                base = f.name
                for ext in [".tiff", ".tif", ".TIFF", ".TIF"]:
                    if base.endswith(ext):
                        base = base[: -len(ext)]
                        break
                p = sid_dir / f"{base}.{fmt}"
                if p.exists():
                    comp_paths.append(p)

            if not comp_paths:
                progress.advance(t3)
                continue

            montage_path = out_montages / f"Montage_{sid}.{fmt}"
            if montage_path.exists() and not args.overwrite:
                progress.advance(t3)
                continue

            build_montage(
                comp_paths,
                montage_path,
                max_side=args.max_montage_side,
                max_pixels=args.max_montage_pixels,
                fmt=fmt,
            )
            progress.advance(t3)

    console.print(f"[bold]Finished writing[/bold] Composites: {out_jpegs}")
    console.print(f"[bold]Finished writing[/bold] Montages: {out_montages}")


if __name__ == "__main__":
    main()
