from pathlib import Path
import argparse

import numpy as np
import tifffile

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect per-plane zero/near-zero stats.")
    p.add_argument("input_tiff", type=Path, help="Path to a TIFF file")
    return p.parse_args()


args = parse_args()
p = args.input_tiff

with tifffile.TiffFile(p) as tif:
    arr = tif.asarray()  # (2, Y, X)

for i in range(arr.shape[0]):
    plane = arr[i]
    total = plane.size
    zeros = np.count_nonzero(plane == 0)
    near_zeros = np.count_nonzero(plane < 5)

    print(f"Plane {i}:")
    print(f"  zeros:       {zeros} ({zeros/total:.2%})")
    print(f"  near-zeros:  {near_zeros} ({near_zeros/total:.2%})")
    print(f"  min / max:   {plane.min()} / {plane.max()}")
    print(f"  mean / std:  {plane.mean():.1f} / {plane.std():.1f}")
