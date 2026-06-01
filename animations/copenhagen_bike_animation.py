"""Self-contained animation: Copenhagen bike-counter hourly net flow.

Same look-and-feel as helsinki_metro_animation: each station is drawn as
a single arrow whose length AND width scale with |dir1 - dir2| for that
hour. Arrow direction follows the local road tangent (OSM); the sign
of net = (+ count - − count) decides forward vs. reverse.

WHY 2014: Copenhagen's open data portal publishes the historical hourly
directional bike-counter archive only for 2005-2014 ("faste cykel-
tællinger" XLSXs). The 2024 service is closed. So this is a Wednesday
in March 2014 — same calendar slot we used for Madrid / Helsinki, just
one decade earlier.

No prerequisite files. The script downloads:
  * 2014 hourly XLSX (Mastra-format report, one workbook, ~1.2 MB).
  * OSM road centerlines via Overpass for the AOI.

Output: copenhagen_bike_<DATE>.mp4 and .gif.
"""
import io, math, os, json, shutil, subprocess, urllib.request
from datetime import date
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from PIL import Image

# --- Config -----------------------------------------------------------------
DATE = "2014-03-12"                                # Wednesday 12 March 2014
DATE_DK = "12.03.2014"                             # Mastra uses dd.mm.yyyy

XLSX_URL = ("https://admin.opendata.dk/dataset/"
            "2d2cfb7c-d93f-43c6-92f7-6f496f0047d3/resource/"
            "ded37b28-998a-40b6-8f86-74b93959c4a9/"
            "download/cykeltaellinger-2014.xlsx")

# 8 x 6 km AOI centred on Rådhuspladsen. Captures the City of Copenhagen
# (København K) plus Frederiksberg, Vesterbro, Østerbro, Nørrebro, Amager.
LAT0, LON0 = 55.6759, 12.5707
SIZE_M = 8000     # we'll keep a wider window for the bike network

DLAT = SIZE_M / 2 / 111_320
DLON = SIZE_M / 2 / (111_320 * math.cos(math.radians(LAT0)))
LAT_MIN, LAT_MAX = LAT0 - DLAT, LAT0 + DLAT
LON_MIN, LON_MAX = LON0 - DLON, LON0 + DLON

# Reproject AOI corners to ETRS89 / UTM 32N (EPSG:25832) — same CRS the
# XLSX uses for Xkoordinat/Ykoordinat.
_aoi = gpd.GeoSeries.from_xy(
    [LON_MIN, LON_MAX, LON_MIN, LON_MAX],
    [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX], crs="EPSG:4326").to_crs(25832)
MINX = float(_aoi.x.min()); MAXX = float(_aoi.x.max())
MINY = float(_aoi.y.min()); MAXY = float(_aoi.y.max())

PX = 1.0
MAX_PX = 6000
PX = max(PX, (MAXX-MINX)/MAX_PX, (MAXY-MINY)/MAX_PX)
W = int(round((MAXX-MINX)/PX)); H = int(round((MAXY-MINY)/PX))
transform = from_origin(MINX, MAXY, PX, PX)
print(f"AOI {MAXX-MINX:.0f} x {MAXY-MINY:.0f} m -> {W} x {H} px @ {PX:.2f} m/px")

CACHE = os.path.expanduser("~/.cache/copenhagen_bike")
os.makedirs(CACHE, exist_ok=True)

OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
ROAD_WIDTH = {
    "motorway": 14, "motorway_link": 6,
    "trunk": 12, "trunk_link": 6,
    "primary": 10, "primary_link": 5,
    "secondary": 8, "secondary_link": 4,
    "tertiary": 7,
    "residential": 6, "unclassified": 6, "living_street": 5,
}

def overpass(query):
    last = None
    for url in OVERPASS_MIRRORS:
        try:
            r = requests.post(url, data={"data": query},
                              headers={"User-Agent": "kbw-cph/1.0"}, timeout=300)
            r.raise_for_status(); return r.json()
        except Exception as e:
            print(f"  overpass mirror {url} failed: {e}"); last = e
    raise last

# --- 1) Road backdrop (motor-vehicle lines, red) ----------------------------
roads_cache = os.path.join(CACHE, "roads.geojson")
if not os.path.exists(roads_cache):
    print("Fetching OSM road centerlines...")
    bbox = f"{LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}"
    osm = overpass(f"""
    [out:json][timeout:300];
    way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street)(_link)?$"]({bbox});
    out geom;
    """)
    feats = []
    for el in osm.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el: continue
        hwy = el.get("tags", {}).get("highway")
        if hwy not in ROAD_WIDTH: continue
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) < 2: continue
        feats.append({"type":"Feature",
                      "geometry":{"type":"LineString","coordinates":coords},
                      "properties":{"highway":hwy}})
    with open(roads_cache, "w") as f:
        json.dump({"type":"FeatureCollection","features":feats}, f)
roads = gpd.read_file(roads_cache).to_crs(25832)
print(f"  road centerlines: {len(roads)}")
roads_buf = roads.copy()
roads_buf["geometry"] = roads_buf.apply(
    lambda r: r.geometry.buffer(ROAD_WIDTH[r["highway"]] / 2), axis=1)
geoms = [g for g in roads_buf.geometry if g is not None and not g.is_empty]
car_mask = rasterize([(g, 1) for g in geoms], out_shape=(H, W),
                     transform=transform, fill=0, dtype="uint8").astype(bool)
base = np.full((H, W, 3), 255, dtype="uint8")
base[car_mask] = (220, 30, 30)

# --- 2) Counter XLSX --------------------------------------------------------
xlsx_path = os.path.join(CACHE, "cykeltaellinger-2014.xlsx")
if not os.path.exists(xlsx_path):
    print(f"Fetching 2014 hourly XLSX...")
    urllib.request.urlretrieve(XLSX_URL, xlsx_path)

print("Reading XLSX...")
df = pd.read_excel(xlsx_path, sheet_name=0, header=10)
# Header looks like: Vej-Id, Vejnavn, Spor, (UTM32), (UTM32).1, Dato,
# kl.00-01, kl.01-02, ..., kl.23-24.
df = df.rename(columns={df.columns[0]:"vej_id", df.columns[1]:"vejnavn",
                         df.columns[2]:"spor", df.columns[3]:"x",
                         df.columns[4]:"y", df.columns[5]:"dato"})
hour_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("kl.")]
day = df[df["dato"].astype(str) == DATE_DK].copy()
# Numeric clean.
for c in ["x","y"] + hour_cols:
    day[c] = pd.to_numeric(day[c], errors="coerce")
day = day.dropna(subset=["vej_id","x","y"])
day["dir"] = day["vej_id"].astype(str).str.strip().str[-1]
day["station_key"] = day["vej_id"].astype(str).str.rstrip(" -+T").str.strip()
print(f"  rows on {DATE}: {len(day)}")

# Build pairs: for each station_key with both + and -, take their hourly arrays.
pairs = []
for key, grp in day.groupby("station_key"):
    has_p = grp[grp["dir"] == "+"]
    has_m = grp[grp["dir"] == "-"]
    if len(has_p) == 0 or len(has_m) == 0: continue
    p = has_p.iloc[0]; m = has_m.iloc[0]
    # Average the two UTM positions; they're usually identical.
    xm = (p["x"] + m["x"]) / 2; ym = (p["y"] + m["y"]) / 2
    pairs.append({"name": p["vejnavn"], "x": xm, "y": ym,
                  "plus_hours": [int(p[c]) for c in hour_cols],
                  "minus_hours": [int(m[c]) for c in hour_cols]})
print(f"  paired stations on {DATE}: {len(pairs)}")

# Filter pairs to AOI and compute pixel positions.
pairs = [p for p in pairs if MINX <= p["x"] < MAXX and MINY <= p["y"] < MAXY]
print(f"  pairs inside AOI: {len(pairs)}")
for p in pairs:
    p["px"] = (p["x"] - MINX) / (MAXX - MINX) * W
    p["py"] = (MAXY - p["y"]) / (MAXY - MINY) * H

# --- 2b) Direction vectors via nearest road tangent -------------------------
from shapely.geometry import Point
def tangent_at(point):
    dists = roads.geometry.distance(point)
    i = dists.idxmin()
    line = roads.geometry.loc[i]
    if line.is_empty: return (1.0, 0.0)
    p_proj = line.project(point)
    a = line.interpolate(p_proj)
    b = line.interpolate(min(p_proj + 50, line.length))
    dx, dy = b.x - a.x, b.y - a.y
    n = math.hypot(dx, dy)
    return (dx/n, dy/n) if n > 0 else (1.0, 0.0)

for p in pairs:
    p["tangent"] = tangent_at(Point(p["x"], p["y"]))

# --- 3) Render frames -------------------------------------------------------
SCALE = 0.20
small = np.array(Image.fromarray(base).resize(
    (int(W*SCALE), int(H*SCALE)), Image.LANCZOS))
sH, sW, _ = small.shape

# Peak |net| over the day.
maxnet = 1
for p in pairs:
    for h in range(24):
        maxnet = max(maxnet, abs(p["plus_hours"][h] - p["minus_hours"][h]))
print(f"  peak |net| hourly count: {maxnet}")

MIN_LEN, MAX_LEN = 6, 48       # 1.5x baseline
MIN_W,  MAX_W  = 2.25, 12      # 1.5x baseline
def arrow_dims(n):
    if n <= 0: return 0, 0, 0, 0
    s = math.sqrt(n / maxnet)
    L = MIN_LEN + (MAX_LEN - MIN_LEN) * s
    w = MIN_W  + (MAX_W  - MIN_W ) * s
    return L, w, w * 2.4, max(4.0, L * 0.45)

frame_dir = os.path.join(CACHE, f"frames_{DATE}")
os.makedirs(frame_dir, exist_ok=True)
for fname in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, fname))

for h in range(24):
    fig, ax = plt.subplots(figsize=(sW/100, sH/100), dpi=100)
    ax.imshow(small); ax.set_xlim(0, sW); ax.set_ylim(sH, 0); ax.axis("off")
    t_plus = t_minus = 0
    for p in pairs:
        plus_n = p["plus_hours"][h]; minus_n = p["minus_hours"][h]
        net = plus_n - minus_n
        t_plus += plus_n; t_minus += minus_n
        if net == 0: continue
        sign = 1 if net > 0 else -1
        dx, dy = p["tangent"]
        dy_screen = -dy        # image y flipped vs world y
        L, w, hw, hl = arrow_dims(abs(net))
        x = p["px"] * SCALE; y = p["py"] * SCALE
        ax.add_patch(FancyArrow(
            x, y, sign*dx*L, sign*dy_screen*L,
            width=w, head_width=hw, head_length=hl,
            length_includes_head=True,
            facecolor=(0.10, 0.50, 0.95, 0.90),
            edgecolor=(0.0, 0.20, 0.55, 0.95), linewidth=0.7))
    ax.text(10, 30, f"Copenhagen bike counters — net flow per station — {DATE}  {h:02d}:00",
            color="white", fontsize=14, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    ax.text(10, sH-30,
            f"dir +: {t_plus:>5,}   dir −: {t_minus:>5,}   net: {t_plus-t_minus:+,}",
            color="white", fontsize=11,
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    fig.savefig(f"{frame_dir}/{h:02d}.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
print(f"  rendered 24 frames -> {frame_dir}")

# --- 4) Encode --------------------------------------------------------------
out_mp4 = f"copenhagen_bike_{DATE}.mp4"
out_gif = f"copenhagen_bike_{DATE}.gif"
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
