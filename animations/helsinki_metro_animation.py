"""Self-contained animation: vehicle counts on national-road TMS stations
across the Helsinki / Espoo / Vantaa metropolitan area, over one day.

Coverage: Kehä I/II/III ring roads + the radial motorways
(Vt 1 / E18 to Turku, Vt 3 / E12 to Tampere, Vt 4 / E75 to Lahti,
Vt 7 / E18 to Porvoo, Vt 25 to Hanko, Kt 45 / Tuusulanväylä, etc.).

Data sources, all anonymous:
  * TMS station inventory:
        https://tie.digitraffic.fi/api/tms/v1/stations
  * Per-station daily raw LAM CSV (one row per vehicle passage):
        https://tie.digitraffic.fi/api/tms/v1/history/raw/lamraw_{id}_{YY}_{DDD}.csv
        (DDD = ordinal day-of-year, NOT zero-padded.)
  * Road network for the backdrop: OpenStreetMap via Overpass.

Output: helsinki_metro_<DATE>.mp4 and .gif.
"""
import io, math, os, csv, json, shutil, subprocess, urllib.request
from datetime import date
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import shape
from rasterio.transform import from_origin
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# --- Config -----------------------------------------------------------------
DATE = "2024-03-13"     # Sunday — match the other animations in this folder.
d_obj = date.fromisoformat(DATE)
YY = d_obj.strftime("%y")
DDD = (d_obj - date(d_obj.year, 1, 1)).days + 1       # 1-based day-of-year

# WGS84 bbox for Helsinki + Espoo + Vantaa metro area.
LAT_MIN, LAT_MAX = 60.150, 60.330
LON_MIN, LON_MAX = 24.550, 25.150

# Reproject to ETRS-TM35FIN (EPSG:3067) — Finland's national metric grid.
_aoi = gpd.GeoSeries.from_xy(
    [LON_MIN, LON_MAX, LON_MIN, LON_MAX],
    [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX], crs="EPSG:4326").to_crs(3067)
MINX = float(_aoi.x.min()); MAXX = float(_aoi.x.max())
MINY = float(_aoi.y.min()); MAXY = float(_aoi.y.max())

PX = 1.0
MAX_PX = 6000
PX = max(PX, (MAXX-MINX)/MAX_PX, (MAXY-MINY)/MAX_PX)
W = int(round((MAXX-MINX)/PX)); H = int(round((MAXY-MINY)/PX))
transform = from_origin(MINX, MAXY, PX, PX)
print(f"AOI {MAXX-MINX:.0f} x {MAXY-MINY:.0f} m -> {W} x {H} px @ {PX:.2f} m/px")

CACHE = os.path.expanduser("~/.cache/helsinki_metro")
os.makedirs(CACHE, exist_ok=True)

OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
# Buffer widths per highway class (m, total width).
ROAD_WIDTH = {
    "motorway": 14, "motorway_link": 6,
    "trunk": 12, "trunk_link": 6,
    "primary": 10, "primary_link": 5,
    "secondary": 8, "secondary_link": 4,
    "tertiary": 7,
}

def overpass(query):
    last = None
    for url in OVERPASS_MIRRORS:
        try:
            r = requests.post(url, data={"data": query},
                              headers={"User-Agent": "kbw-hel-metro/1.0"},
                              timeout=300)
            r.raise_for_status(); return r.json()
        except Exception as e:
            print(f"  overpass mirror {url} failed: {e}"); last = e
    raise last

# --- 1) Road backdrop (motorway / trunk / primary / secondary / tertiary) --
roads_cache = os.path.join(CACHE, "roads.geojson")
if not os.path.exists(roads_cache):
    print("Fetching OSM major roads (Overpass)...")
    bbox = f"{LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}"
    osm = overpass(f"""
    [out:json][timeout:300];
    way["highway"~"^(motorway|trunk|primary|secondary|tertiary)(_link)?$"]({bbox});
    out geom;
    """)
    feats = []
    for el in osm.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el: continue
        hwy = el.get("tags", {}).get("highway")
        if hwy not in ROAD_WIDTH: continue
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) < 2: continue
        feats.append({"type": "Feature",
                       "geometry": {"type": "LineString", "coordinates": coords},
                       "properties": {"highway": hwy}})
    with open(roads_cache, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)

roads = gpd.read_file(roads_cache).to_crs(3067)
print(f"  road centerlines: {len(roads)}")
roads["geometry"] = roads.apply(
    lambda r: r.geometry.buffer(ROAD_WIDTH[r["highway"]] / 2), axis=1)
geoms = [g for g in roads.geometry if g is not None and not g.is_empty]
car_mask = rasterize([(g, 1) for g in geoms], out_shape=(H, W),
                     transform=transform, fill=0, dtype="uint8").astype(bool)
base = np.full((H, W, 3), 255, dtype="uint8")
base[car_mask] = (220, 30, 30)

# --- 2) TMS station inventory ----------------------------------------------
stations_path = os.path.join(CACHE, "stations.geojson")
if not os.path.exists(stations_path):
    print("Fetching TMS station inventory...")
    r = requests.get("https://tie.digitraffic.fi/api/tms/v1/stations",
                     headers={"Accept-Encoding": "gzip"}, timeout=120)
    r.raise_for_status()
    with open(stations_path, "wb") as f: f.write(r.content)
sf = gpd.read_file(stations_path)
sf = sf[sf.geometry.notnull()].copy()
sf = sf.to_crs(3067)
sf["x_m"] = sf.geometry.x; sf["y_m"] = sf.geometry.y
in_aoi = sf[(sf.x_m >= MINX)&(sf.x_m < MAXX)&(sf.y_m >= MINY)&(sf.y_m < MAXY)].copy()
in_aoi["px"] = (in_aoi.x_m - MINX) / (MAXX - MINX) * W
in_aoi["py"] = (MAXY - in_aoi.y_m) / (MAXY - MINY) * H
print(f"  TMS stations in AOI: {len(in_aoi)} of {len(sf)}")

# --- 3) Per-station, per-direction hourly counts ----------------------------
# LAM raw schema columns 0..15:
#   0 pistetunnus, 1 vuosi, 2 vuorokausi, 3 tunti, 4 min, 5 sek, 6 ms,
#   7 pituus(m), 8 ajoneuvoluokka(1..7), 9 ajosuunta(1|2), 10 kaista,
#   11 nopeus, 12 faulty, 13 kokonaisaika, 14 jonoaika, 15 viim.
# Direction lives in column 9, NOT 8 (column 8 is the vehicle class and is
# almost always 1=passenger car, which made it look like a binary signal).
print(f"Fetching raw LAM CSVs (day {DDD} of 20{YY})...")
hourly = {1: {}, 2: {}}                        # hourly[direction][sid] = [24 ints]
missing = 0
for _, s in in_aoi.iterrows():
    sid = int(s["tmsNumber"])
    fp = os.path.join(CACHE, f"lamraw_{sid}_{YY}_{DDD}.csv")
    if not os.path.exists(fp):
        url = f"https://tie.digitraffic.fi/api/tms/v1/history/raw/lamraw_{sid}_{YY}_{DDD}.csv"
        try:
            r = requests.get(url, timeout=120)
            if r.status_code != 200: missing += 1; continue
            with open(fp, "wb") as f: f.write(r.content)
        except Exception:
            missing += 1; continue
    try:
        d1 = [0] * 24; d2 = [0] * 24
        with open(fp) as f:
            for row in csv.reader(f, delimiter=";"):
                if len(row) < 10: continue
                h = int(row[3])
                d = int(row[9])
                if 0 <= h < 24:
                    (d1 if d == 1 else d2)[h] += 1
        hourly[1][sid] = d1
        hourly[2][sid] = d2
    except Exception:
        missing += 1
print(f"  fetched data for {len(hourly[1])} stations, {missing} missing")

# --- 3b) Direction vectors via nearest OSM road -----------------------------
# We need a 2D unit vector for each station so the arrows point along the
# road. Snap each station to the nearest road centerline; tangent at that
# point = arrow axis for direction 1; -tangent = direction 2. Which way
# direction 1 actually flows depends on the station's install; we display
# both arrows from the station outward in opposite directions so the
# *relative* magnitude tells the inbound/outbound story regardless.
from shapely.ops import nearest_points
roads_raw = gpd.read_file(roads_cache).to_crs(3067)   # lines, not buffers
def tangent_at(point, road_lines):
    """Return a unit (dx, dy) along the nearest road segment near point."""
    # Find the road whose geometry is nearest to the point.
    dists = road_lines.distance(point)
    near_idx = dists.idxmin()
    line = road_lines.loc[near_idx]
    if line.is_empty: return (1.0, 0.0)
    proj = line.interpolate(line.project(point))
    # Step a few metres along the line and take the chord.
    p2 = line.interpolate(min(line.project(point) + 50, line.length))
    dx, dy = p2.x - proj.x, p2.y - proj.y
    n = math.hypot(dx, dy)
    return (dx/n, dy/n) if n > 0 else (1.0, 0.0)

in_aoi["tangent"] = [tangent_at(g, roads_raw.geometry) for g in in_aoi.geometry]

# --- 4) Render frames -------------------------------------------------------
SCALE = 0.18
small = np.array(Image.fromarray(base).resize(
    (int(W*SCALE), int(H*SCALE)), Image.LANCZOS))
sH, sW, _ = small.shape

# Peak ABSOLUTE NET flow across the day, used to scale arrow size (both
# length AND width). We combine the two directions at each station into a
# single vector: net = (count_dir1 - count_dir2) along the road tangent.
# Sign of net decides which way the arrow points; |net| sets its size.
maxnet = 1
for _, s in in_aoi.iterrows():
    sid = int(s["tmsNumber"])
    d1 = hourly[1].get(sid, [0]*24); d2 = hourly[2].get(sid, [0]*24)
    for h in range(24):
        maxnet = max(maxnet, abs(d1[h] - d2[h]))
print(f"  peak net (|dir1 - dir2|) hourly count: {maxnet}")

frame_dir = os.path.join(CACHE, f"frames_{DATE}")
os.makedirs(frame_dir, exist_ok=True)
for fname in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, fname))

# Arrow geometry — both length and width scale with sqrt(|net|/maxnet).
# We draw with FancyArrowPatch so we can set width directly (the simple
# ax.annotate("", arrowprops=...) approach can't scale width per arrow).
from matplotlib.patches import FancyArrow
MIN_LEN, MAX_LEN = 6, 48       # pixels on the small image (1.5x baseline)
MIN_W,  MAX_W  = 2.25, 12      # arrow stem width (1.5x baseline)

def arrow_dims(n):
    """Return (length, stem-width, head-width, head-length) for |net| = n."""
    if n <= 0: return 0, 0, 0, 0
    s = math.sqrt(n / maxnet)
    L = MIN_LEN + (MAX_LEN - MIN_LEN) * s
    w = MIN_W  + (MAX_W  - MIN_W ) * s
    return L, w, w * 2.4, max(4.0, L * 0.45)

for h in range(24):
    fig, ax = plt.subplots(figsize=(sW/100, sH/100), dpi=100)
    ax.imshow(small); ax.set_xlim(0, sW); ax.set_ylim(sH, 0); ax.axis("off")
    for _, s in in_aoi.iterrows():
        sid = int(s["tmsNumber"])
        d1 = hourly[1].get(sid, [0]*24)[h]
        d2 = hourly[2].get(sid, [0]*24)[h]
        net = d1 - d2
        if net == 0: continue
        L, w, hw, hl = arrow_dims(abs(net))
        # Direction sign — positive net -> along tangent, negative -> opposite.
        sign = 1 if net > 0 else -1
        dx, dy = s["tangent"]
        dy_screen = -dy        # image y is flipped vs world
        x = s["px"] * SCALE; y = s["py"] * SCALE
        ax.add_patch(FancyArrow(
            x, y, sign*dx*L, sign*dy_screen*L,
            width=w, head_width=hw, head_length=hl,
            length_includes_head=True,
            facecolor=(0.95, 0.40, 0.05, 0.90),
            edgecolor=(0.55, 0.15, 0.0, 0.95), linewidth=0.8))
    ax.text(10, 30,
            f"Helsinki metro TMS — net flow per station — {DATE}  {h:02d}:00",
            color="white", fontsize=14, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    t1 = sum(hourly[1].get(int(s["tmsNumber"]), [0]*24)[h] for _, s in in_aoi.iterrows())
    t2 = sum(hourly[2].get(int(s["tmsNumber"]), [0]*24)[h] for _, s in in_aoi.iterrows())
    net_total = t1 - t2
    ax.text(10, sH-30,
            f"dir 1: {t1:>7,}   dir 2: {t2:>7,}   net (1-2): {net_total:+,}",
            color="white", fontsize=11,
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    fig.savefig(f"{frame_dir}/{h:02d}.png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
print(f"  rendered 24 frames -> {frame_dir}")

# --- 5) Encode MP4 + GIF ---------------------------------------------------
out_mp4 = f"helsinki_metro_{DATE}.mp4"
out_gif = f"helsinki_metro_{DATE}.gif"
if shutil.which("ffmpeg"):
    subprocess.run(["ffmpeg", "-y", "-framerate", "3",
                    "-i", f"{frame_dir}/%02d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_mp4],
                   check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print(f"Saved {out_mp4}")

frames = [Image.open(f"{frame_dir}/{h:02d}.png").convert(
    "P", palette=Image.ADAPTIVE, colors=128) for h in range(24)]
frames[0].save(out_gif, save_all=True, append_images=frames[1:],
               duration=333, loop=0, optimize=True, disposal=2)
print(f"Saved {out_gif}")
