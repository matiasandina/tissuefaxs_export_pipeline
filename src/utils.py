from pathlib import Path
import re
import numpy as np
import tifffile


def region_key(path: Path) -> str:
    """
    Extract the region identifier prefix up to 'Region XXX' from a filename.
    """
    m = re.match(r"(.*_Region \d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot extract region key from {path.name}")
    return m.group(1)


def load_plane(path: Path) -> np.ndarray:
    """
    Load a single 2D TIFF plane from disk.
    """
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
    if arr.ndim != 2:
        raise ValueError(f"{path.name} is not 2D, got shape {arr.shape}")
    return arr

def load_planes(paths) -> np.ndarray:
    """
    Load multiple 2D TIFF planes and stack into (C, Y, X).
    """
    planes = []
    for p in paths:
        with tifffile.TiffFile(p) as tif:
            arr = tif.asarray()
        if arr.ndim != 2:
            raise ValueError(f"{p.name} is not 2D, got shape {arr.shape}")
        planes.append(arr)
    return np.stack(planes, axis=0)  # (C, Y, X)

def build_pyramid(img: np.ndarray, levels: int = 4):
    """
    Build a simple 2x downsampled pyramid from a (C, Y, X) image.
    Returns list of arrays, level 0 = full resolution.
    """
    pyr = [img]
    for _ in range(1, levels):
        prev = pyr[-1]
        down = prev[:, ::2, ::2]
        pyr.append(down)
    return pyr

def write_pyramidal_ome_tiff(
    out_path: Path,
    pyramid,
    physical_size_xy,
    channel_names=("mDAPI", "mTxRed"),
    tile=(512, 512),
):
    """
    Write a pyramidal OME-TIFF with basic OME metadata and SubIFDs.
    """
    base = pyramid[0]
    c, y, x = base.shape

    with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
        tif.write(
            base,
            subifds=len(pyramid) - 1,
            tile=tile,
            compression=None,
            photometric="minisblack",
            metadata={
                "axes": "CYX",
                "Channel": {"Name": list(channel_names)},
                "PhysicalSizeX": physical_size_xy,
                "PhysicalSizeY": physical_size_xy,
            },
        )

        for level in pyramid[1:]:
            tif.write(
                level,
                tile=tile,
                compression=None,
            )


def write_pyramidal_qupath_tiff(
    out_path: Path,
    pyramid,
    tile=(512, 512),
):
    """
    Write a pyramidal TIFF that QuPath will read correctly:
    - planar channels
    - SubIFDs for pyramid
    - NO OME metadata
    """

    base = pyramid[0]          # (C, Y, X)
    c, y, x = base.shape

    with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
        # write full-res level
        tif.write(
            base,
            subifds=len(pyramid) - 1,
            tile=tile,
            compression=None,
            photometric="minisblack",
            planarconfig="separate",  # THIS is important
            metadata=None,            # critical: no OME
            description=None,
        )

        # write pyramid levels as SubIFDs
        for level in pyramid[1:]:
            tif.write(
                level,
                tile=tile,
                compression=None,
                photometric="minisblack",
                planarconfig="separate",
                metadata=None,
            )

def get_physical_pixel_size(path: Path) -> float:
    """
    Read PhysicalSizeX/Y from OME metadata and assert they match.
    """
    with tifffile.TiffFile(path) as tif:
        ome = tif.ome_metadata
    if ome is None:
        raise ValueError(f"No OME metadata in {path.name}")

    # minimal XML parse, no heavy deps
    import xml.etree.ElementTree as ET
    root = ET.fromstring(ome)

    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    pixels = root.find(".//ome:Pixels", ns)

    px = float(pixels.attrib["PhysicalSizeX"])
    py = float(pixels.attrib["PhysicalSizeY"])

    if abs(px - py) > 1e-6:
        raise ValueError("PhysicalSizeX != PhysicalSizeY, unexpected")

    return px

def normalize_and_rotate_region(
    channel_paths,
    channel_names,
    out_path,
    pyramid_levels=4,
):
    """
    Stack per-channel planes, rotate 180 deg, build pyramid, and write OME-TIFF.
    """
    if len(channel_paths) != len(channel_names):
        raise ValueError("paths and channel_names must have same length")

    stack = load_planes(channel_paths)
    rotated = np.flip(stack, axis=(1, 2))

    pyramid = build_pyramid(rotated, levels=pyramid_levels)

    physical_size_xy = get_physical_pixel_size(channel_paths[0])

    write_pyramidal_ome_tiff(
        out_path,
        pyramid,
        physical_size_xy=physical_size_xy,
        channel_names=channel_names,
    )
