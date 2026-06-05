"""Self-contained animation: Melbourne pedestrian-counter hourly flow
with DIRECTIONAL arrows, one frame per hour for a single day.

Why Melbourne: the City of Melbourne's Pedestrian Counting System is
the densest open pedestrian-sensor network we know of — 136 permanent
sensors, hourly counts, split by `direction_1` and `direction_2`
(strings like "North"/"South"). Coverage is concentrated in the CBD,
which is exactly the area we want to visualise.

Data sources
------------
* Aerial basemap: ESRI World Imagery REST export (EPSG:28355), the
  same orthophoto source the static city overlays use. No road
  geometry is drawn — the photo itself shows the streets.
* Sensor locations: data.melbourne.vic.gov.au dataset
  `pedestrian-counting-system-sensor-locations`
  (location_id, lat, lon, direction_1, direction_2, status).
* Hourly counts: dataset
  `pedestrian-counting-system-monthly-counts-per-hour`
  (location_id, sensing_date, hourday, direction_1, direction_2,
   pedestriancount).

Visual encoding
---------------
At each active sensor the two opposing direction counts are collapsed
into a SINGLE net-flow arrow — pointing the way of the busier
direction, with length AND width both scaled by |direction_1 -
direction_2|, exactly like the Helsinki metro animation. Bearing is
taken from the *string* direction ("North", "NorthEast", ...);
sensors whose direction strings are not in our compass map (rare
custom labels) are skipped.

Output: melbourne_pedestrian_<DATE>.mp4 and .gif.
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
from PIL import Image

# --- Config -----------------------------------------------------------------
DATE = "2026-05-28"      # Thursday — weekday commuter pattern

# AOI: Melbourne CBD ~3.5 x 3 km, expressed in EPSG:28355 (MGA Zone 55,
# a metric projection appropriate for Victoria).
LON_MIN, LON_MAX = 144.940, 144.985
LAT_MIN, LAT_MAX = -37.830, -37.795
to_local = Transformer.from_crs("EPSG:4326", "EPSG:28355", always_xy=True)
to_wgs = Transformer.from_crs("EPSG:28355", "EPSG:4326", always_xy=True)
MINX, MINY = to_local.transform(LON_MIN, LAT_MIN)
MAXX, MAXY = to_local.transform(LON_MAX, LAT_MAX)
# Round to nice metric values.
MINX, MAXX = round(MINX, -2), round(MAXX, -2)
MINY, MAXY = round(MINY, -2), round(MAXY, -2)
PX = 2.0
W = int((MAXX - MINX) / PX); H = int((MAXY - MINY) / PX)
transform = from_origin(MINX, MAXY, PX, PX)
print(f"AOI {MAXX-MINX:.0f} x {MAXY-MINY:.0f} m -> {W} x {H} px @ {PX} m/px")

API = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
SENSORS_URL = f"{API}/pedestrian-counting-system-sensor-locations/records"
COUNTS_URL = f"{API}/pedestrian-counting-system-monthly-counts-per-hour/records"
ORTHO = ("https://services.arcgisonline.com/arcgis/rest/services/"
         "World_Imagery/MapServer/export")

CACHE = os.path.expanduser("~/.cache/melbourne_ped")
os.makedirs(CACHE, exist_ok=True)


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
                "bboxSR": "28355", "imageSR": "28355",
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

# --- 2) Sensor locations ---------------------------------------------------
sensors_cache = os.path.join(CACHE, "sensors.json")
if not os.path.exists(sensors_cache):
    print("Fetching sensor locations...")
    all_sensors = []
    offset = 0
    while True:
        url = f"{SENSORS_URL}?limit=100&offset={offset}"
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        all_sensors.extend(page["results"])
        if len(page["results"]) < 100: break
        offset += 100
    with open(sensors_cache, "w") as f: json.dump(all_sensors, f)
with open(sensors_cache) as f:
    all_sensors = json.load(f)
print(f"  sensors total: {len(all_sensors)}")

# Compass-direction → unit vector in display pixel space (y grows down).
BEARING = {
    "north":     (0.0, -1.0),
    "south":     (0.0,  1.0),
    "east":      (1.0,  0.0),
    "west":      (-1.0, 0.0),
    "northeast": (math.sqrt(0.5), -math.sqrt(0.5)),
    "northwest": (-math.sqrt(0.5), -math.sqrt(0.5)),
    "southeast": (math.sqrt(0.5),  math.sqrt(0.5)),
    "southwest": (-math.sqrt(0.5), math.sqrt(0.5)),
}
def bearing(s: str) -> tuple[float, float] | None:
    if not s: return None
    return BEARING.get(s.strip().lower().replace(" ", ""))

# Keep only sensors that are active, inside the AOI, and have both
# direction labels that we can parse.
sensors: dict[int, dict] = {}
for s in all_sensors:
    if s.get("status") != "A": continue
    lon, lat = s.get("longitude"), s.get("latitude")
    if lon is None or lat is None: continue
    x, y = to_local.transform(lon, lat)
    px = (x - MINX) / (MAXX - MINX) * W
    py = (MAXY - y) / (MAXY - MINY) * H
    if not (0 <= px < W and 0 <= py < H): continue
    b1 = bearing(s.get("direction_1")); b2 = bearing(s.get("direction_2"))
    if b1 is None or b2 is None: continue
    sensors[s["location_id"]] = {
        "px": px, "py": py, "b1": b1, "b2": b2,
        "name": s.get("sensor_description") or s.get("sensor_name"),
    }
print(f"  active sensors with parseable bearings inside AOI: {len(sensors)}")

# --- 3) Hourly counts for DATE ---------------------------------------------
counts_cache = os.path.join(CACHE, f"counts_{DATE}.json")
if not os.path.exists(counts_cache):
    print(f"Fetching hourly counts for {DATE}...")
    all_rows = []
    offset = 0
    while True:
        params = [
            ("where", f"sensing_date = date'{DATE}'"),
            ("limit", 100), ("offset", offset),
            ("select", "location_id,hourday,direction_1,direction_2,pedestriancount"),
        ]
        url = f"{COUNTS_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=120) as r:
            page = json.load(r)
        all_rows.extend(page["results"])
        if len(page["results"]) < 100: break
        offset += 100
        if offset >= 10000:
            print("  hit offset cap"); break
    with open(counts_cache, "w") as f: json.dump(all_rows, f)
with open(counts_cache) as f:
    rows = json.load(f)
print(f"  count rows: {len(rows)}")

hour_data: dict[int, list[dict]] = {h: [] for h in range(24)}
for r in rows:
    lid = r["location_id"]
    if lid not in sensors: continue
    h = int(r["hourday"])
    hour_data[h].append({
        "px": sensors[lid]["px"], "py": sensors[lid]["py"],
        "b1": sensors[lid]["b1"], "b2": sensors[lid]["b2"],
        "n1": int(r.get("direction_1") or 0),
        "n2": int(r.get("direction_2") or 0),
    })

# Peak NET magnitude |direction_1 - direction_2| across the day — scales the
# arrows (both length and width), exactly like the Helsinki metro animation.
maxc = max((max((abs(d["n1"] - d["n2"]) for d in lst), default=0)
            for lst in hour_data.values()), default=1) or 1
print(f"  peak net (|dir1 - dir2|) hourly count: {maxc}")

# --- 3b) Hourly air temperature (Open-Meteo, 2 m) --------------------------
# Free, no key, historical-capable. We query the CBD centroid and key the
# 24 hourly °C values by hour-of-day for the per-frame readout.
CBD_LAT, CBD_LON = -37.8136, 144.9631
temp_cache = os.path.join(CACHE, f"temp_{DATE}.json")
if not os.path.exists(temp_cache):
    print("Fetching hourly temperature (Open-Meteo)...")
    params = urllib.parse.urlencode({
        "latitude": CBD_LAT, "longitude": CBD_LON,
        "hourly": "temperature_2m",
        "start_date": DATE, "end_date": DATE,
        "timezone": "Australia/Melbourne"})
    with urllib.request.urlopen(
            f"https://api.open-meteo.com/v1/forecast?{params}", timeout=60) as r:
        payload = json.load(r)
    with open(temp_cache, "w") as f: json.dump(payload, f)
with open(temp_cache) as f:
    tpayload = json.load(f)
_th = tpayload.get("hourly", {})
temp_by_hour = {int(t[11:13]): v for t, v in
                zip(_th.get("time", []), _th.get("temperature_2m", []))
                if v is not None}
print(f"  hourly temps: {len(temp_by_hour)}/24  "
      f"({min(temp_by_hour.values()):.1f}..{max(temp_by_hour.values()):.1f} °C)")

# --- 4) Render frames -------------------------------------------------------
SCALE = 0.30
small = np.array(Image.fromarray(base).resize(
    (int(W * SCALE), int(H * SCALE)), Image.LANCZOS))
sH, sW, _ = small.shape

# Arrow geometry — both length and width scale with sqrt(|net|/maxc), drawn
# with FancyArrow so the stem width can vary per arrow (Helsinki-metro scale).
from matplotlib.patches import FancyArrow
MIN_LEN, MAX_LEN = 6, 48        # pixels on the small image
MIN_W,  MAX_W  = 2.25, 12       # arrow stem width
def arrow_dims(n):
    """Return (length, stem-width, head-width, head-length) for |net| = n."""
    if n <= 0: return 0, 0, 0, 0
    s = math.sqrt(n / maxc)
    L = MIN_LEN + (MAX_LEN - MIN_LEN) * s
    w = MIN_W  + (MAX_W  - MIN_W ) * s
    return L, w, w * 2.4, max(4.0, L * 0.45)

frame_dir = os.path.join(CACHE, f"frames_{DATE}")
os.makedirs(frame_dir, exist_ok=True)
for fname in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, fname))

for h in range(24):
    fig, ax = plt.subplots(figsize=(sW / 100, sH / 100), dpi=100)
    ax.imshow(small); ax.set_xlim(0, sW); ax.set_ylim(sH, 0); ax.axis("off")
    for d in hour_data[h]:
        x = d["px"] * SCALE; y = d["py"] * SCALE
        # Collapse the (direction_1, direction_2) pair into a SINGLE net-flow
        # arrow: it points the busier way and its size tracks |n1 - n2|.
        n1, n2 = d["n1"], d["n2"]
        net = n1 - n2
        if net == 0: continue
        bx, by = d["b1"] if net > 0 else d["b2"]
        L, w, hw, hl = arrow_dims(abs(net))
        if L <= 0: continue
        ax.add_patch(FancyArrow(
            x, y, bx * L, by * L,
            width=w, head_width=hw, head_length=hl,
            length_includes_head=True,
            facecolor=(0.90, 0.20, 0.20, 0.92),
            edgecolor=(0.55, 0.05, 0.05, 0.95), linewidth=0.6))
    ax.text(10, 28, f"Melbourne pedestrian flow — {DATE}  {h:02d}:00",
            color="white", fontsize=15, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    if h in temp_by_hour:
        ax.text(sW - 10, 28, f"{temp_by_hour[h]:.1f} °C",
                color="white", fontsize=14, fontweight="bold",
                ha="right", va="center",
                bbox=dict(facecolor="black", alpha=0.6, pad=4))
    tot = sum(d["n1"] + d["n2"] for d in hour_data[h])
    ax.text(10, sH - 15,
            f"{tot:,} pedestrians across {len(hour_data[h])} active sensors this hour",
            color="white", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    fig.savefig(f"{frame_dir}/{h:02d}.png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
print(f"  rendered 24 frames -> {frame_dir}")

# --- 5) Encode MP4 + GIF ---------------------------------------------------
out_mp4 = f"melbourne_pedestrian_{DATE}.mp4"
out_gif = f"melbourne_pedestrian_{DATE}.gif"
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
