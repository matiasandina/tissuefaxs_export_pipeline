from pathlib import Path
import numpy as np
import tifffile
import warnings

from .utils import (
    build_pyramid,
    write_pyramidal_ome_tiff,
    get_physical_pixel_size,
    write_pyramidal_qupath_tiff
)

warnings.filterwarnings(
    "ignore",
    message="OME series contains invalid TiffData index",
)


def rotate_channels_file(
    in_path: Path,
    out_path: Path,
    channel_names=("mDAPI", "mTxRed"),
    pyramid_levels=4,
    output = "qupath"
):
    with tifffile.TiffFile(in_path) as tif:
        arr = tif.asarray()  # (C, Y, X)

    if arr.ndim != 3:
        raise ValueError(f"{in_path.name} expected (C,Y,X), got {arr.shape}")

    rotated = np.flip(arr, axis=(1, 2))  # 180°

    pyramid = build_pyramid(rotated, levels=pyramid_levels)

    physical_size_xy = get_physical_pixel_size(in_path)

    if output == "qupath":
        write_pyramidal_qupath_tiff(
        out_path,
        pyramid
        )
    else:
        write_pyramidal_ome_tiff(
            out_path,
            pyramid,
            physical_size_xy=physical_size_xy,
            channel_names=channel_names,
        )
