"""Self-contained animation: Bologna 'spire' loop-detector counts over one
day, rendered on top of a freshly-built map showing only the city's
motor-vehicle road surface (OSM, buffered centerlines).

No prerequisite files — this script downloads everything it needs:
  * OSM road centerlines (via Overpass) -> rasterised buffer
  * Spire sensor positions + hourly counts for the chosen day
    (Comune di Bologna OpenData, dataset
     rilevazione-flusso-veicoli-tramite-spire-anno-YYYY).

Output: bologna_spire_<DATE>.mp4 and .gif.
"""
import io, math, os, json, csv, urllib.request, shutil, subprocess
from collections import defaultdict
import numpy as np
import requests
import geopandas as gpd
from shapely.geometry import shape
from rasterio.transform import from_origin
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# --- Config -----------------------------------------------------------------
DATE = "2024-03-13"                                       # Sunday 10 Mar 2024
YEAR = DATE[:4]
SPIRE_DS = f"rilevazione-flusso-veicoli-tramite-spire-anno-{YEAR}"
EXPORT = ("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/"
          f"datasets/{SPIRE_DS}/exports/geojson")

# 4 x 4 km centred on Piazza Maggiore; reproject to UTM 32N (EPSG:32632).
LAT0, LON0 = 44.4938, 11.3426
SIZE_M = 4000

DLAT = SIZE_M / 2 / 111_320
DLON = SIZE_M / 2 / (111_320 * math.cos(math.radians(LAT0)))
lat_min, lat_max = LAT0 - DLAT, LAT0 + DLAT
lon_min, lon_max = LON0 - DLON, LON0 + DLON
_aoi = gpd.GeoSeries.from_xy(
    [lon_min, lon_max, lon_min, lon_max],
    [lat_min, lat_min, lat_max, lat_max], crs="EPSG:4326").to_crs(32632)
MINX = float(_aoi.x.min()); MAXX = float(_aoi.x.max())
MINY = float(_aoi.y.min()); MAXY = float(_aoi.y.max())

PX = 0.5
MAX_PX = 6000
PX = max(PX, (MAXX-MINX)/MAX_PX, (MAXY-MINY)/MAX_PX)
W = int(round((MAXX-MINX)/PX)); H = int(round((MAXY-MINY)/PX))
transform = from_origin(MINX, MAXY, PX, PX)
print(f"AOI {MAXX-MINX:.0f} x {MAXY-MINY:.0f} m -> {W} x {H} px @ {PX:.3f} m/px")

CACHE = os.path.expanduser("~/.cache/bologna_spire")
os.makedirs(CACHE, exist_ok=True)

# --- 1) Car road mask from OSM (buffered centerlines) -----------------------
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
ROAD_WIDTH = {
    "motorway": 14, "trunk": 12, "primary": 10, "secondary": 8,
    "tertiary": 7, "unclassified": 6, "residential": 6,
    "living_street": 5, "service": 4, "track": 3,
}

def overpass(query):
    last = None
    for url in OVERPASS_MIRRORS:
        try:
            r = requests.post(url, data={"data": query},
                              headers={"User-Agent": "kbw-bologna-anim/1.0"},
                              timeout=300)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  overpass mirror {url} failed: {e}")
            last = e
    raise last

osm_cache = os.path.join(CACHE, "roads.geojson")
if not os.path.exists(osm_cache):
    print("Fetching OSM road centerlines...")
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    osm = overpass(f"""
    [out:json][timeout:300];
    way["highway"]({bbox});
    out geom;
    """)
    feats = []
    for el in osm.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el: continue
        tags = el.get("tags", {})
        hwy = tags.get("highway")
        if hwy not in ROAD_WIDTH: continue
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) < 2: continue
        feats.append({"type": "Feature",
                       "geometry": {"type": "LineString", "coordinates": coords},
                       "properties": {"highway": hwy}})
    with open(osm_cache, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)

roads = gpd.read_file(osm_cache).to_crs(32632)
print(f"  road centerlines: {len(roads)}")
roads["geometry"] = roads.apply(
    lambda r: r.geometry.buffer(ROAD_WIDTH[r["highway"]] / 2), axis=1)
geoms = [g for g in roads.geometry if g is not None and not g.is_empty]
car_mask = rasterize([(g, 1) for g in geoms], out_shape=(H, W),
                     transform=transform, fill=0, dtype="uint8").astype(bool)
base = np.full((H, W, 3), 255, dtype="uint8")
base[car_mask] = (220, 30, 30)

# --- 2) Spire sensors + counts for the chosen day ---------------------------
spire_cache = os.path.join(CACHE, f"spire_{DATE}.geojson")
if not os.path.exists(spire_cache):
    print(f"Fetching spire counts for {DATE} ...")
    r = requests.get(EXPORT, params={"where": f"data = date'{DATE}'"}, timeout=600)
    r.raise_for_status()
    with open(spire_cache, "wb") as f:
        f.write(r.content)

spire = gpd.read_file(spire_cache)
if spire.crs is None: spire = spire.set_crs("EPSG:4326")
spire = spire.to_crs(32632)
# Keep only sensors inside the AOI.
in_aoi = spire[(spire.geometry.x >= MINX) & (spire.geometry.x < MAXX) &
               (spire.geometry.y >= MINY) & (spire.geometry.y < MAXY)].copy()
in_aoi["px"] = (in_aoi.geometry.x - MINX) / (MAXX - MINX) * W
in_aoi["py"] = (MAXY - in_aoi.geometry.y) / (MAXY - MINY) * H
# Each spira reports one row per day; aggregate by id_uni so duplicated
# spire IDs at the same location combine.
hour_cols = [f"{h:02d}_00_{h+1:02d}_00" if h < 23 else "23_00_24_00"
             for h in range(24)]
# Some rows may lack hourly fields; pandas will fill with NaN -> 0.
in_aoi[hour_cols] = in_aoi[hour_cols].fillna(0).astype(int)
print(f"  spire sensors inside AOI: {len(in_aoi)}")

# --- 3) Render frames -------------------------------------------------------
SCALE = 0.20
small = np.array(Image.fromarray(base).resize(
    (int(W*SCALE), int(H*SCALE)), Image.LANCZOS))
sH, sW, _ = small.shape

maxc = max(in_aoi[hc].max() for hc in hour_cols)
print(f"  peak single-spira hourly count: {maxc}")

frame_dir = os.path.join(CACHE, f"frames_{DATE}")
os.makedirs(frame_dir, exist_ok=True)
for fname in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, fname))

for h in range(24):
    hc = hour_cols[h]
    fig, ax = plt.subplots(figsize=(sW/100, sH/100), dpi=100)
    ax.imshow(small); ax.set_xlim(0, sW); ax.set_ylim(sH, 0); ax.axis("off")
    for _, s in in_aoi.iterrows():
        n = int(s[hc])
        if n <= 0: continue
        r = 3 + 25 * math.sqrt(n / maxc)
        ax.add_patch(mpatches.Circle(
            (s["px"] * SCALE, s["py"] * SCALE), radius=r,
            facecolor=(1, 0.5, 0, 0.65),
            edgecolor=(0.6, 0.2, 0, 0.95), linewidth=1.0))
    ax.text(10, 30, f"Bologna 'spire' counts — {DATE}  {h:02d}:00",
            color="white", fontsize=16, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    total = int(in_aoi[hc].sum())
    ax.text(10, sH-15,
            f"{total:,} vehicles past {len(in_aoi)} sensors this hour",
            color="white", fontsize=12,
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    fig.savefig(f"{frame_dir}/{h:02d}.png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
print(f"  rendered 24 frames -> {frame_dir}")

# --- 4) Encode MP4 (small) + GIF -------------------------------------------
out_mp4 = f"bologna_spire_{DATE}.mp4"
out_gif = f"bologna_spire_{DATE}.gif"
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
