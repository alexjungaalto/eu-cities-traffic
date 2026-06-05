"""On-disk cache shared by the city-map scripts.

Goal: an interrupted run should resume from whatever was already
downloaded instead of re-fetching. Every expensive download (ESRI/WMS
orthophoto tiles, the assembled ortho mosaic, Overpass/WFS/REST vector
responses) is written to a per-script cache directory and looked up
there first on the next run.

Cache root:  ~/.cache/eu-cities-maps/<script-name>/
  tiles_<W>x<H>_<tile_px>/<ix>_<iy>.png   one orthophoto tile
  ortho_<W>x<H>.npy                       assembled RGB mosaic
  <key>.json                              cached JSON response
  <key>.bin                               cached raw bytes (GeoJSON/GML/…)

All writes are atomic (write to <path>.tmp, then os.replace) so a
process killed mid-write never leaves a half-file that poisons the
cache. A corrupt mosaic is detected on load and simply rebuilt from the
tile cache, so no download is repeated unnecessarily.
"""
import os
import numpy as np

ROOT = os.path.expanduser("~/.cache/eu-cities-maps")


def cache_dir(name, *sub):
    d = os.path.join(ROOT, name, *sub)
    os.makedirs(d, exist_ok=True)
    return d


def _atomic_write(path, data, mode="wb"):
    tmp = path + ".tmp"
    with open(tmp, mode) as f:
        f.write(data)
    os.replace(tmp, path)


def tile_bytes(name, W, H, tile_px, ix, iy, fetch):
    """Return the PNG bytes for orthophoto tile (ix, iy), fetching with
    `fetch()` only if it is not already cached on disk."""
    d = cache_dir(name, f"tiles_{W}x{H}_{tile_px}")
    p = os.path.join(d, f"{ix}_{iy}.png")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        with open(p, "rb") as f:
            return f.read()
    data = fetch()
    _atomic_write(p, data)
    return data


def cached_array(name, key, build):
    """Return a cached numpy array (e.g. the ortho mosaic), or build it.

    `key` is used as the filename stem (e.g. f"ortho_{W}x{H}"). If a
    cached .npy exists and is readable it is returned without calling
    `build`; a corrupt file is ignored and rebuilt."""
    p = os.path.join(cache_dir(name), key + ".npy")
    if os.path.exists(p):
        try:
            arr = np.load(p)
            print(f"  [cache] {key}.npy")
            return arr
        except Exception:
            print(f"  [cache] {key}.npy unreadable — rebuilding")
    arr = build()
    tmp = p + ".tmp"
    np.save(tmp, arr)            # np.save appends .npy -> tmp + '.npy'
    os.replace(tmp + ".npy", p)
    return arr


def cached_json(name, key, build):
    """Return a cached JSON-serialisable object, or build and cache it."""
    import json
    p = os.path.join(cache_dir(name), key + ".json")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        try:
            with open(p) as f:
                obj = json.load(f)
            print(f"  [cache] {key}.json")
            return obj
        except Exception:
            print(f"  [cache] {key}.json unreadable — refetching")
    obj = build()
    _atomic_write(p, json.dumps(obj).encode(), mode="wb")
    return obj


def cached_bytes(name, key, build):
    """Return cached raw bytes (GeoJSON/GML/etc.), or build and cache them.
    `key` may include an extension; if it has none, '.bin' is appended."""
    if "." not in os.path.basename(key):
        key = key + ".bin"
    p = os.path.join(cache_dir(name), key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        print(f"  [cache] {key}")
        with open(p, "rb") as f:
            return f.read()
    data = build()
    _atomic_write(p, data)
    return data
