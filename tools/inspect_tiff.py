from pathlib import Path
import tifffile as t
import json
import hashlib

def pixel_hash(page):
    # hash raw pixel bytes only
    return hashlib.sha256(page.asarray().tobytes()).hexdigest()

def inspect(path):
    with t.TiffFile(path) as tf:
        page = tf.pages[0]
        tags = page.tags

        def get(tag):
            return tags[tag].value if tag in tags else None

        return {
            "path": str(path),
            "shape": page.shape,
            "dtype": str(page.dtype),
            "compression": str(page.compression),
            "tiled": page.is_tiled,
            "x_resolution": get("XResolution"),
            "y_resolution": get("YResolution"),
            "resolution_unit": get("ResolutionUnit"),
            "image_description": get("ImageDescription"),
            "pixel_hash": pixel_hash(page),
        }

if __name__ == "__main__":
    import sys, glob
    from pathlib import Path

    paths = []
    for arg in sys.argv[1:]:
        matches = glob.glob(arg)
        if matches:
            paths.extend(matches)
        else:
            # also allow passing an explicit file path
            if Path(arg).exists():
                paths.append(arg)

    if not paths:
        raise SystemExit("No input files matched. Example: python inspect_tiff.py /.../original/*.tiff")

    out = {}
    for p in map(Path, paths):
        out[p.name] = inspect(p)

    import json
    print(json.dumps(out, indent=2, default=str))