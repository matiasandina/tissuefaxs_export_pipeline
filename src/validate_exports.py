from pathlib import Path
from collections import defaultdict
import tifffile
import numpy as np

from .utils import region_key, get_physical_pixel_size

CHANNEL_SUFFIXES = ["mDAPI", "mTxRed"]  # extend later


def inspect_tiff(path: Path):
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
        ome = tif.ome_metadata

    info = {
        "path": str(path),
        "shape": arr.shape,
        "ndim": arr.ndim,
        "is_rgb": arr.ndim == 3 and arr.shape[-1] == 3,
        "is_stack": arr.ndim == 3 and arr.shape[0] <= 5,
        "physical_size_xy": None,
    }

    try:
        info["physical_size_xy"] = get_physical_pixel_size(path)
    except Exception as e:
        info["physical_size_error"] = str(e)

    return info


def validate_region(region, files):
    report = {
        "region": region,
        "files": {},
        "errors": [],
        "warnings": [],
        "ok": True,
    }

    shapes = set()
    phys_sizes = set()
    channel_files = []

    for p in files:
        info = inspect_tiff(p)
        report["files"][p.name] = info

        if info.get("is_rgb"):
            continue

        shapes.add(info["shape"][-2:])

        if "physical_size_xy" in info and info["physical_size_xy"] is not None:
            phys_sizes.add(info["physical_size_xy"])

        channel_files.append(p)

    if len(channel_files) == 0:
        report["errors"].append("No scientific channels found")
        report["ok"] = False

    if len(shapes) > 1:
        report["errors"].append(f"Inconsistent spatial shapes: {shapes}")
        report["ok"] = False

    if len(phys_sizes) > 1:
        report["errors"].append(f"Inconsistent physical pixel sizes: {phys_sizes}")
        report["ok"] = False

    return report


def validate_directory(input_dir: Path):
    regions = defaultdict(list)

    for p in input_dir.glob("*.tiff"):
        try:
            rk = region_key(p)
        except ValueError:
            continue
        regions[rk].append(p)

    results = {}
    for rk, files in regions.items():
        results[rk] = validate_region(rk, files)

    return results
