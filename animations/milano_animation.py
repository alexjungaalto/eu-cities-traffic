"""Self-contained animation: Area C gate traffic over one day, rendered
on top of a freshly-built map showing only Milano's motor-vehicle
surface (DBT class A010101_area_circolazione_veicolare).

No prerequisite files — this script downloads everything it needs:
  * the DBT car polygons from Milano's ArcGIS FeatureServer
  * the 42 Area C gate locations (GeoJSON)
  * the 30-min transit counts for the chosen day (ZIPed CSV)

Output: milano_areac_<DATE>.mp4 and .gif.
"""
import io, math, os, zipfile, json, csv, urllib.request
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
DATE = "2024-03-10"            # Sunday 10 March 2024
MONTH_DS_ID = "ds2730"          # March 2024 dataset
MONTH_ZIP = ("https://dati.comune.milano.it/dataset/"
             "fc1ed5c4-4582-42e3-b2f2-0387a1079959/resource/"
             "d8da6d46-a153-4099-869b-0d65c6a92a36/download/"
             "ds2730_ingressi_areac_2024_03.zip")
GATES_URL = ("https://dati.comune.milano.it/dataset/"
             "4cad1605-8225-4ecd-9b82-868b3af453e5/resource/"
             "fa8fcc31-1722-4a50-a0ae-ce7b9c0d0361/download/"
             "ingressi_areac_varchi.geojson")
FS = ("https://geoportale.comune.milano.it/arcgis/rest/services/"
      "Cartografie_Vettoriali/DBT_2012_Milano_RDN2008_UTM32/FeatureServer")
CAR_LAYER_ID = 31

# 4 x 4 km AOI centred on Piazza del Duomo, EPSG:25832 (UTM 32N).
MINX, MINY = 512800, 5032600
MAXX, MAXY = 516800, 5036600
PX = 0.5
MAX_PX = 6000
PX = max(PX, (MAXX-MINX)/MAX_PX, (MAXY-MINY)/MAX_PX)
W = int(round((MAXX-MINX)/PX)); H = int(round((MAXY-MINY)/PX))
transform = from_origin(MINX, MAXY, PX, PX)
print(f"AOI {MAXX-MINX:.0f} x {MAXY-MINY:.0f} m -> {W} x {H} px @ {PX:.3f} m/px")

CACHE = os.path.expanduser("~/.cache/milano_areac")
os.makedirs(CACHE, exist_ok=True)

def cache_path(name):
    return os.path.join(CACHE, name)

# --- 1) DBT car polygons (paginated by quadtree, ArcGIS caps at 2000 feats) -
def fetch_car_polygons():
    cache = cache_path("car_polys.geojson")
    if os.path.exists(cache):
        return gpd.read_file(cache)
    feats = []; stack = [(MINX, MINY, MAXX, MAXY)]
    while stack:
        bb = stack.pop()
        r = requests.get(f"{FS}/{CAR_LAYER_ID}/query", params={
            "where": "1=1",
            "geometry": f"{bb[0]},{bb[1]},{bb[2]},{bb[3]}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "25832", "outSR": "25832",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "OBJECTID", "f": "geojson",
        }, timeout=600)
        r.raise_for_status()
        gj = r.json()
        chunk = gj.get("features", [])
        exceeded = gj.get("exceededTransferLimit") or \
                   gj.get("properties", {}).get("exceededTransferLimit")
        if exceeded and (bb[2]-bb[0]) > 50:
            mx = (bb[0]+bb[2])/2; my = (bb[1]+bb[3])/2
            stack += [(bb[0], bb[1], mx, my), (mx, bb[1], bb[2], my),
                      (bb[0], my, mx, bb[3]), (mx, my, bb[2], bb[3])]
        else:
            feats.extend(chunk)
    seen, uniq = set(), []
    for f in feats:
        fid = f.get("id") or f.get("properties", {}).get("OBJECTID")
        if fid in seen: continue
        seen.add(fid); uniq.append(f)
    gdf = gpd.GeoDataFrame.from_features(uniq, crs="EPSG:25832")
    gdf.to_file(cache, driver="GeoJSON")
    return gdf

print("Fetching DBT car polygons...")
cars = fetch_car_polygons()
print(f"  {len(cars)} polygons")

# Rasterise car polygons -> boolean mask -> red on white base image.
geoms = [g for g in cars.geometry if g is not None and not g.is_empty]
car_mask = rasterize([(g, 1) for g in geoms], out_shape=(H, W),
                     transform=transform, fill=0, dtype="uint8").astype(bool)
base = np.full((H, W, 3), 255, dtype="uint8")
base[car_mask] = (220, 30, 30)

# --- 2) Gate locations ------------------------------------------------------
gates_path = cache_path("varchi.geojson")
if not os.path.exists(gates_path):
    print("Fetching gate locations...")
    urllib.request.urlretrieve(GATES_URL, gates_path)
gates = gpd.read_file(gates_path).to_crs(25832)
def to_px(x, y):
    return ((x-MINX)/(MAXX-MINX)*W, (MAXY-y)/(MAXY-MINY)*H)
gates["px"], gates["py"] = zip(*[to_px(g.x, g.y) for g in gates.geometry])
in_aoi = gates[(gates.px >= 0)&(gates.px < W)&(gates.py >= 0)&(gates.py < H)]
print(f"  gates inside AOI: {len(in_aoi)} of {len(gates)}")

# --- 3) Transit counts for the chosen day ----------------------------------
zip_path = cache_path(MONTH_ZIP.rsplit("/", 1)[-1])
if not os.path.exists(zip_path):
    print("Fetching transit counts (zip)...")
    urllib.request.urlretrieve(MONTH_ZIP, zip_path)
with zipfile.ZipFile(zip_path) as z:
    csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
    csv_path = cache_path(csv_name)
    if not os.path.exists(csv_path):
        z.extractall(CACHE)
counts = defaultdict(lambda: defaultdict(int))
with open(csv_path) as f:
    for row in csv.DictReader(f):
        if row["dataora"].startswith(DATE):
            counts[row["dataora"][:16]][int(row["id_varco"])] += int(row["numero_transiti"])
slots = sorted(counts.keys())
print(f"  time slots: {len(slots)}  ({slots[0]} .. {slots[-1]})")

# --- 4) Render frames -------------------------------------------------------
SCALE = 0.20                                   # 6000 -> 1200 px wide
small = np.array(Image.fromarray(base).resize(
    (int(W*SCALE), int(H*SCALE)), Image.LANCZOS))
sH, sW, _ = small.shape
maxc = max(max(d.values()) for d in counts.values())
print(f"  peak gate count in any 30-min slot: {maxc}")

frame_dir = cache_path(f"frames_{DATE}")
os.makedirs(frame_dir, exist_ok=True)
for fname in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, fname))

for i, ts in enumerate(slots):
    fig, ax = plt.subplots(figsize=(sW/100, sH/100), dpi=100)
    ax.imshow(small); ax.set_xlim(0, sW); ax.set_ylim(sH, 0); ax.axis("off")
    for _, g in in_aoi.iterrows():
        n = counts[ts].get(int(g["id_amat"]), 0)
        r = 3 + 35 * math.sqrt(n / maxc)
        ax.add_patch(mpatches.Circle(
            (g.px * SCALE, g.py * SCALE), radius=r,
            facecolor=(1, 0.5, 0, 0.7),
            edgecolor=(0.6, 0.2, 0, 0.95), linewidth=1.5))
    ax.text(10, 30, f"Milano Area C — {DATE}  {ts[11:]}",
            color="white", fontsize=16, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    total = sum(counts[ts].values())
    ax.text(10, sH-15, f"{total:,} vehicles entering Area C this 30-min slot",
            color="white", fontsize=12,
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    fig.savefig(f"{frame_dir}/{i:03d}.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
print(f"  rendered {len(slots)} frames -> {frame_dir}")

# --- 5) Encode MP4 (small) + GIF (fallback for static viewers) -------------
out_mp4 = f"milano_areac_{DATE}.mp4"
out_gif = f"milano_areac_{DATE}.gif"

# Need ffmpeg for the mp4; fall back to GIF-only if missing.
import shutil, subprocess
if shutil.which("ffmpeg"):
    subprocess.run(["ffmpeg", "-y", "-framerate", "5",
                    "-i", f"{frame_dir}/%03d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_mp4],
                   check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print(f"Saved {out_mp4}")
else:
    print("(ffmpeg not on PATH — skipping mp4)")

# Always also write a palettised GIF.
frames = [Image.open(f"{frame_dir}/{i:03d}.png").convert(
    "P", palette=Image.ADAPTIVE, colors=128) for i in range(len(slots))]
frames[0].save(out_gif, save_all=True, append_images=frames[1:],
               duration=200, loop=0, optimize=True, disposal=2)
print(f"Saved {out_gif}")
