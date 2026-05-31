"""Self-contained animation: Helsinki bike-counter hourly flow over one
day, on top of a freshly-built map showing only the city's motor-vehicle
road surface (YLRE class `ajorata_alue`).

Why bikes for Helsinki, not cars: Helsinki publishes hourly cyclist
counts from 16 permanent counter stations (since 2014) but only annual
KAVL aggregates for cars — no city-wide hourly car counter dump.

No prerequisite files. The script downloads:
  * YLRE carriageway polygons via Helsinki's WFS (avoindata layer).
  * The bike counter locations CSV (Latin-1 encoded — handled below).
  * The full hourly count XLSX (~11 MB) for all years and stations.

Output: helsinki_bike_<DATE>.mp4 and .gif.
"""
import io, math, os, csv, urllib.request, shutil, subprocess
from collections import defaultdict
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
DATE = "2024-03-13"                                  # Sunday — match Milano/Bologna

# 4 x 4 km AOI centred on Helsinki Central Station (Rautatieasema)
# in EPSG:3879 (ETRS-GK25, Helsinki's local cadastral CRS).
MINX, MINY = 25494800, 6671200
MAXX, MAXY = 25498800, 6675200
PX = 0.5
MAX_PX = 6000
PX = max(PX, (MAXX-MINX)/MAX_PX, (MAXY-MINY)/MAX_PX)
W = int(round((MAXX-MINX)/PX)); H = int(round((MAXY-MINY)/PX))
transform = from_origin(MINX, MAXY, PX, PX)
print(f"AOI {MAXX-MINX:.0f} x {MAXY-MINY:.0f} m -> {W} x {H} px @ {PX:.3f} m/px")

WFS = "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"
SITES_URL = "https://www.hel.fi/static/avoindata/kymp/pyoralaskentapisteiden_sijainnit.csv"
XLSX_URL  = "https://www.hel.fi/static/avoindata/kymp/Helsingin_pyorailijamaarat.xlsx"

CACHE = os.path.expanduser("~/.cache/helsinki_bike")
os.makedirs(CACHE, exist_ok=True)

# --- 1) YLRE car polygons ---------------------------------------------------
ylre_cache = os.path.join(CACHE, "ylre_car.geojson")
if not os.path.exists(ylre_cache):
    print("Fetching YLRE carriageway polygons...")
    r = requests.get(WFS, params={
        "service":"WFS","version":"2.0.0","request":"GetFeature",
        "typeNames":"avoindata:YLRE_Katu_ja_viherosat_ajorata_alue",
        "srsName":"EPSG:3879",
        "bbox":f"{MINX},{MINY},{MAXX},{MAXY},EPSG:3879",
        "outputFormat":"application/json","count":50000}, timeout=600)
    r.raise_for_status()
    with open(ylre_cache, "wb") as f: f.write(r.content)
cars = gpd.read_file(ylre_cache)
print(f"  YLRE car polygons: {len(cars)}")

geoms = [g for g in cars.geometry if g is not None and not g.is_empty]
car_mask = rasterize([(g, 1) for g in geoms], out_shape=(H, W),
                     transform=transform, fill=0, dtype="uint8").astype(bool)
base = np.full((H, W, 3), 255, dtype="uint8")
base[car_mask] = (220, 30, 30)

# --- 2) Counter locations ---------------------------------------------------
sites_path = os.path.join(CACHE, "sites.csv")
if not os.path.exists(sites_path):
    print("Fetching counter locations...")
    urllib.request.urlretrieve(SITES_URL, sites_path)
# The CSV is Latin-1 (mojibake-prone). Parse explicitly and split fields by ';'.
sites = []
with open(sites_path, encoding="latin-1") as f:
    rdr = csv.reader(f, delimiter=";")
    header = next(rdr)
    for row in rdr:
        name = row[0].strip()
        x = float(row[1].replace(",", "."))
        y = float(row[2].replace(",", "."))
        sites.append({"name": name, "x": x, "y": y})
sites_df = pd.DataFrame(sites)
sites_df["px"] = (sites_df.x - MINX) / (MAXX - MINX) * W
sites_df["py"] = (MAXY - sites_df.y) / (MAXY - MINY) * H
in_aoi = sites_df[(sites_df.px >= 0)&(sites_df.px < W)&
                  (sites_df.py >= 0)&(sites_df.py < H)].copy()
print(f"  counter stations inside AOI: {len(in_aoi)} of {len(sites_df)}")

# --- 3) Hourly counts XLSX -------------------------------------------------
xlsx_path = os.path.join(CACHE, "counts.xlsx")
if not os.path.exists(xlsx_path):
    print(f"Fetching hourly counts XLSX (~11 MB)...")
    urllib.request.urlretrieve(XLSX_URL, xlsx_path)

print("Reading XLSX...")
counts_df = pd.read_excel(xlsx_path, sheet_name=0, parse_dates=["Päivämäärä_aika"])
day = counts_df[counts_df["Päivämäärä"].astype(str).str.startswith(DATE)].copy()
if len(day) == 0:
    # Some sheets store the date column as datetime, not string — fall back.
    day = counts_df[counts_df["Päivämäärä_aika"].dt.strftime("%Y-%m-%d") == DATE].copy()
day["hour"] = pd.to_datetime(day["Päivämäärä_aika"]).dt.hour
print(f"  rows for {DATE}: {len(day)} (expected 24)")

# Map CSV station names to XLSX column names. The two datasets use
# slightly different spellings; we match by normalising whitespace and
# scandic characters, then keep stations that exist on both sides.
def norm(s):
    return (s or "").strip().lower().replace("ä","a").replace("ö","o").replace(" ","")
xlsx_cols = {norm(c): c for c in counts_df.columns if c not in
             ("Päivämäärä_aika", "Päivämäärä", "Aika")}
in_aoi["col"] = in_aoi["name"].map(lambda n: xlsx_cols.get(norm(n)))
matched = in_aoi.dropna(subset=["col"])
print(f"  AOI stations matched to XLSX columns: {len(matched)} of {len(in_aoi)}")
if len(matched) == 0:
    print("WARNING: no station/column matches — adjust the norm() function.")

# Pull per-hour counts; some stations have NaN if they were offline.
hour_to_counts = {}
for h in range(24):
    row = day[day.hour == h]
    if len(row) == 0: continue
    vals = {}
    for _, s in matched.iterrows():
        try: vals[s["name"]] = int(row[s["col"]].iloc[0])
        except Exception: vals[s["name"]] = 0
    hour_to_counts[h] = vals

# --- 4) Render frames -------------------------------------------------------
SCALE = 0.20
small = np.array(Image.fromarray(base).resize(
    (int(W*SCALE), int(H*SCALE)), Image.LANCZOS))
sH, sW, _ = small.shape

# Find peak count across the day so the radius scale stays consistent.
maxc = max((max(v.values()) if v else 0) for v in hour_to_counts.values()) or 1
print(f"  peak single-station hourly count: {maxc}")

frame_dir = os.path.join(CACHE, f"frames_{DATE}")
os.makedirs(frame_dir, exist_ok=True)
for fname in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, fname))

for h in range(24):
    vals = hour_to_counts.get(h, {})
    fig, ax = plt.subplots(figsize=(sW/100, sH/100), dpi=100)
    ax.imshow(small); ax.set_xlim(0, sW); ax.set_ylim(sH, 0); ax.axis("off")
    for _, s in matched.iterrows():
        n = vals.get(s["name"], 0)
        if n <= 0: continue
        r = 4 + 30 * math.sqrt(n / maxc)
        ax.add_patch(mpatches.Circle(
            (s["px"] * SCALE, s["py"] * SCALE), radius=r,
            facecolor=(0.1, 0.5, 1.0, 0.7),
            edgecolor=(0.0, 0.2, 0.6, 0.95), linewidth=1.2))
    ax.text(10, 30, f"Helsinki bike counters — {DATE}  {h:02d}:00",
            color="white", fontsize=16, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    total = sum(vals.values())
    ax.text(10, sH-15,
            f"{total:,} cyclists past {len(matched)} sensors this hour",
            color="white", fontsize=12,
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    fig.savefig(f"{frame_dir}/{h:02d}.png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
print(f"  rendered 24 frames -> {frame_dir}")

# --- 5) Encode MP4 + GIF ---------------------------------------------------
out_mp4 = f"helsinki_bike_{DATE}.mp4"
out_gif = f"helsinki_bike_{DATE}.gif"
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
