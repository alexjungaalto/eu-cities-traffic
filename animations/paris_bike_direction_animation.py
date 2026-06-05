"""Self-contained animation: Paris bike-counter hourly flow with
DIRECTIONAL arrows, one frame per hour for a single day.

Why Paris and why directional: opendata.paris.fr publishes ~100+
permanent bike Eco-Counters with **per-direction** sensors. Each
counter site has two sensors (e.g. "boulevard Diderot E-O" and
"...O-E") that share an `id` but differ in `id_compteur`. Plotting
both as arrows shows commuter asymmetry — morning rush points
inbound, evening rush outbound — which a single-blob count map hides.

Data sources
------------
* Aerial basemap: ESRI World Imagery REST export (EPSG:2154), the
  same orthophoto source the static city overlays use. No road
  geometry is drawn — the photo itself shows the streets.
* Counter records: opendata.paris.fr dataset
  `comptage-velo-donnees-compteurs` — hourly volumes since 2018.

Direction parsing
-----------------
`nom_compteur` ends with a compass suffix like "E-O", "SO-NE",
"NE-SO". We treat the suffix as `FROM-TO` (Eco-Counter's common
convention): the arrow points toward the second token. A site that
records both ways has two sensors (one per direction); we combine
them into a SINGLE net-flow arrow per site — pointing the way of the
busier direction, length/width scaled by |dir_a - dir_b| — exactly
like the Helsinki metro animation. Sites with only one sensor keep a
single arrow whose size tracks that sensor's count.

Output: paris_bike_direction_<DATE>.mp4 and .gif.
"""
import io, math, os, urllib.parse, urllib.request, json, shutil, subprocess
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# --- Config -----------------------------------------------------------------
DATE = "2026-05-28"      # Thursday, typical weekday — strong commuter asymmetry

# AOI: ~6 x 5 km box centred roughly on Châtelet / Île de la Cité,
# expressed in Lambert-93 (EPSG:2154 — France's metric national CRS).
MINX, MINY = 647500, 6857500
MAXX, MAXY = 653500, 6862500
PX = 2.0
W = int((MAXX - MINX) / PX); H = int((MAXY - MINY) / PX)
transform = from_origin(MINX, MAXY, PX, PX)
print(f"AOI {MAXX-MINX:.0f} x {MAXY-MINY:.0f} m -> {W} x {H} px @ {PX} m/px")

OPENDATA = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"
COUNTS_API = f"{OPENDATA}/comptage-velo-donnees-compteurs/records"
ORTHO = ("https://services.arcgisonline.com/arcgis/rest/services/"
         "World_Imagery/MapServer/export")

CACHE = os.path.expanduser("~/.cache/paris_bike_dir")
os.makedirs(CACHE, exist_ok=True)
to_local = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)


def ortho_image(tile_px=2048):
    """Mosaic ESRI World Imagery over the AOI at the raster grid resolution.
    Tiled so each REST request stays within the service's size limits."""
    out = np.zeros((H, W, 3), "uint8")
    nx = math.ceil(W / tile_px); ny = math.ceil(H / tile_px)
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * tile_px; y0 = iy * tile_px
            tw = min(tile_px, W - x0); th = min(tile_px, H - y0)
            tminx = MINX + x0 * PX;  tmaxx = tminx + tw * PX
            tmaxy = MAXY - y0 * PX;  tminy = tmaxy - th * PX
            r = requests.get(ORTHO, params={
                "bbox": f"{tminx},{tminy},{tmaxx},{tmaxy}",
                "bboxSR": "2154", "imageSR": "2154",
                "size": f"{tw},{th}", "format": "png", "f": "image"}, timeout=600)
            r.raise_for_status()
            out[y0:y0+th, x0:x0+tw] = np.array(
                Image.open(io.BytesIO(r.content)).convert("RGB"))
            print(f"  ortho tile {iy*nx+ix+1}/{nx*ny}")
    return out

# --- 1) Aerial basemap (no road geometry) ----------------------------------
# The orthophoto itself shows the streets, so we draw no road polygons.
# Cached as a .npy so re-runs skip the tile fetch.
ortho_cache = os.path.join(CACHE, f"ortho_{W}x{H}.npy")
if os.path.exists(ortho_cache):
    base = np.load(ortho_cache)
else:
    print("Fetching orthophoto (ESRI World Imagery)...")
    base = ortho_image()
    np.save(ortho_cache, base)

# --- 2) Counter records for DATE -------------------------------------------
# The OpenData API is paged at 100 rows; we use `where=` to filter by date
# and paginate with `offset` until we run out.
counts_cache = os.path.join(CACHE, f"counts_{DATE}.json")
if not os.path.exists(counts_cache):
    print(f"Fetching counter records for {DATE}...")
    all_rows = []
    offset = 0
    while True:
        next_date = (pd.Timestamp(DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        params = [
            ("select", "id,id_compteur,nom_compteur,sum_counts,date,coordinates"),
            ("where", f"date >= '{DATE}' and date < '{next_date}'"),
            ("limit", 100),
            ("offset", offset),
        ]
        url = f"{COUNTS_API}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=120) as r:
            payload = json.load(r)
        chunk = payload["results"]
        all_rows.extend(chunk)
        if len(chunk) < 100:
            break
        offset += 100
        if offset >= 10000:  # OpenData caps at offset=10000
            print("  hit 10k-offset cap; truncating")
            break
    with open(counts_cache, "w") as fh: json.dump(all_rows, fh)
with open(counts_cache) as fh:
    rows = json.load(fh)
print(f"  records fetched: {len(rows)}")

# --- 3) Build per-counter metadata + per-hour values -----------------------
# Compass suffix -> unit vector (dx, dy) in *display* pixel space.
# Pixel y grows downward, so southward dy is positive.
BEARING = {
    "N":  (0.0, -1.0),
    "S":  (0.0,  1.0),
    "E":  (1.0,  0.0),
    "O":  (-1.0, 0.0),
    "NE": (math.sqrt(0.5), -math.sqrt(0.5)),
    "NO": (-math.sqrt(0.5), -math.sqrt(0.5)),
    "SE": (math.sqrt(0.5),  math.sqrt(0.5)),
    "SO": (-math.sqrt(0.5), math.sqrt(0.5)),
}
def parse_bearing(name: str) -> tuple[float, float] | None:
    """Return the unit vector the arrow should point along, or None.
    Names look like '27 boulevard Diderot E-O' — last whitespace-token
    is 'E-O', which we interpret as FROM-TO and aim at TO."""
    tail = (name or "").rsplit(" ", 1)[-1]
    if "-" not in tail: return None
    _, to = tail.split("-", 1)
    return BEARING.get(to.strip())

# Map sensor (id_compteur) -> (px, py, bearing, name)
sensors: dict[str, dict] = {}
for r in rows:
    cid = r.get("id_compteur") or r.get("id")
    if cid in sensors: continue
    coords = r.get("coordinates")
    if not coords: continue
    x, y = to_local.transform(coords["lon"], coords["lat"])
    px = (x - MINX) / (MAXX - MINX) * W
    py = (MAXY - y) / (MAXY - MINY) * H
    if not (0 <= px < W and 0 <= py < H):
        continue
    b = parse_bearing(r.get("nom_compteur", ""))
    if b is None:
        continue
    sensors[cid] = {"px": px, "py": py, "bearing": b, "site": r.get("id"),
                    "name": r.get("nom_compteur", "")}
print(f"  sensors with valid bearing inside AOI: {len(sensors)}")

# Per-hour count: sensor -> hour -> count
hour_counts: dict[int, dict[str, int]] = {h: {} for h in range(24)}
for r in rows:
    cid = r.get("id_compteur") or r.get("id")
    if cid not in sensors: continue
    dt = r.get("date", "")
    if not dt or len(dt) < 13: continue
    h = int(dt[11:13])
    hour_counts[h][cid] = hour_counts[h].get(cid, 0) + int(r.get("sum_counts") or 0)

# --- 3b) Group the two directional sensors of each site into one arrow -------
# A counter site (shared `id`) usually has two sensors with opposite bearings.
# We collapse them into a single NET arrow: it points along whichever
# direction carried more bikes this hour, and its length/width scales with
# |count_a - count_b|. Sites with one sensor keep that sensor's raw count as
# the "net". This mirrors the Helsinki metro animation's net-flow arrows.
sites: dict[str, dict] = {}
for cid, s in sensors.items():
    sid = s["site"]
    site = sites.setdefault(sid, {"px": s["px"], "py": s["py"], "dirs": []})
    site["dirs"].append({"cid": cid, "bearing": s["bearing"]})
print(f"  counter sites: {len(sites)} "
      f"({sum(len(v['dirs']) == 2 for v in sites.values())} two-direction)")

def site_net(site: dict, h: int) -> tuple[float, tuple[float, float]]:
    """Return (magnitude, unit-bearing) of the net flow at a site for hour h.
    Two sensors -> signed difference along the busier direction; one sensor
    -> its own count along its bearing."""
    vals = hour_counts.get(h, {})
    dirs = site["dirs"]
    if len(dirs) >= 2:
        a, b = dirs[0], dirs[1]
        ca = vals.get(a["cid"], 0); cb = vals.get(b["cid"], 0)
        net = ca - cb
        if net == 0:
            return 0.0, (0.0, 0.0)
        return abs(net), (a["bearing"] if net > 0 else b["bearing"])
    d = dirs[0]
    return vals.get(d["cid"], 0), d["bearing"]

# Peak net magnitude across the day, used to scale arrow size.
maxc = 1
for site in sites.values():
    for h in range(24):
        maxc = max(maxc, site_net(site, h)[0])
print(f"  peak per-site net hourly count: {maxc}")

# --- 4) Render frames -------------------------------------------------------
SCALE = 0.25
small = np.array(Image.fromarray(base).resize(
    (int(W * SCALE), int(H * SCALE)), Image.LANCZOS))
sH, sW, _ = small.shape

frame_dir = os.path.join(CACHE, f"frames_{DATE}")
os.makedirs(frame_dir, exist_ok=True)
for fname in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, fname))

# Arrow geometry — both length and width scale with sqrt(|net|/maxc), drawn
# with FancyArrow so the stem width can vary per arrow (like Helsinki metro).
from matplotlib.patches import FancyArrow
MIN_LEN, MAX_LEN = 6, 48        # pixels on the small image (Helsinki-metro scale)
MIN_W,  MAX_W  = 2.25, 12       # arrow stem width (Helsinki-metro scale)

def arrow_dims(n):
    """Return (length, stem-width, head-width, head-length) for |net| = n."""
    if n <= 0: return 0, 0, 0, 0
    s = math.sqrt(n / maxc)
    L = MIN_LEN + (MAX_LEN - MIN_LEN) * s
    w = MIN_W  + (MAX_W  - MIN_W ) * s
    return L, w, w * 2.4, max(4.0, L * 0.45)

for h in range(24):
    vals = hour_counts.get(h, {})
    fig, ax = plt.subplots(figsize=(sW / 100, sH / 100), dpi=100)
    ax.imshow(small); ax.set_xlim(0, sW); ax.set_ylim(sH, 0); ax.axis("off")
    for site in sites.values():
        n, (bx, by) = site_net(site, h)
        if n <= 0: continue
        L, w, hw, hl = arrow_dims(n)
        x0, y0 = site["px"] * SCALE, site["py"] * SCALE
        ax.add_patch(FancyArrow(
            x0, y0, bx * L, by * L,
            width=w, head_width=hw, head_length=hl,
            length_includes_head=True,
            facecolor=(0.90, 0.20, 0.20, 0.92),
            edgecolor=(0.55, 0.05, 0.05, 0.95), linewidth=0.6))
    ax.text(10, 30, f"Paris bike counters — net flow — {DATE}  {h:02d}:00",
            color="white", fontsize=16, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    total = sum(vals.values())
    ax.text(10, sH - 15,
            f"{total:,} cyclists past {len(sites)} counter sites this hour",
            color="white", fontsize=11,
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    fig.savefig(f"{frame_dir}/{h:02d}.png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
print(f"  rendered 24 frames -> {frame_dir}")

# --- 5) Encode MP4 + GIF ---------------------------------------------------
out_mp4 = f"paris_bike_direction_{DATE}.mp4"
out_gif = f"paris_bike_direction_{DATE}.gif"
if shutil.which("ffmpeg"):
    subprocess.run(["ffmpeg", "-y", "-framerate", "3",
                    "-i", f"{frame_dir}/%02d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_mp4],
                   check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print(f"Saved {out_mp4}")
else:
    print("(ffmpeg not on PATH — skipping mp4)")

frames = [Image.open(f"{frame_dir}/{h:02d}.png").convert(
    "P", palette=Image.ADAPTIVE, colors=128) for h in range(24)]
frames[0].save(out_gif, save_all=True, append_images=frames[1:],
               duration=333, loop=0, optimize=True, disposal=2)
print(f"Saved {out_gif}")
