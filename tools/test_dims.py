from pathlib import Path
import argparse

import tifffile
import numpy as np

from bioio.writers import OmeTiffWriter
from bioio import BioImage

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a clean OME-TIFF and re-read metadata.")
    p.add_argument("input_tiff", type=Path, help="Path to an input TIFF file")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("test_clean_bioio.tiff"),
        help="Output path for the clean OME-TIFF",
    )
    return p.parse_args()


args = parse_args()
in_file = args.input_tiff
out_file = args.out

# 1. Read pixels only (ignore all metadata)
with tifffile.TiffFile(in_file) as tif:
    data = tif.asarray()

print("raw array shape:", data.shape)

data = np.ascontiguousarray(data)

# 2. Write clean OME-TIFF WITHOUT physical pixel sizes
OmeTiffWriter.save(
    data,
    uri=out_file,
    dim_order="CYX",
    channel_names=["mDAPI", "mTxRed"],
)

print("wrote:", out_file)

# 3. Read back
img = BioImage(out_file)

print("dims:", img.dims)
print("shape:", img.shape)

pixels = img.metadata.images[0].pixels
print("SizeC:", pixels.size_c)
print("SizeT:", pixels.size_t)
print("SizeZ:", pixels.size_z)
print("DimensionOrder:", pixels.dimension_order)
print("PhysicalSizeX:", pixels.physical_size_x)
print("PhysicalSizeY:", pixels.physical_size_y)
