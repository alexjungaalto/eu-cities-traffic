"""Self-contained animation: Helsinki pedestrian-counter hourly flow over
one day, on top of a freshly-built map of the city's light-traffic
surfaces (YLRE class `kevytliikenne_alue` — sidewalks, mixed
pedestrian/bike areas, plazas) shown in blue.

Sister script to `helsinki_bike_animation.py`. Same AOI, same encoding.

Data sources
------------
* YLRE light-traffic polygons via Helsinki avoindata WFS.
* Pedestrian counts via the Lidotiku REST API (`lidotiku.api.hel.fi`),
  filtering `source=EcoCounter` and `vehicletype=pedestrian`. As of 2026
  the city operates ~14 permanent pedestrian Eco-Counters, most of them
  in the central AOI used here (Aleksanterinkatu, Eteläesplanadi,
  Fredrikinkatu, Forumin tunneli, Vanha Ylioppilastalo, Erottaja, ...).

Output: helsinki_pedestrian_<DATE>.mp4 and .gif.
"""
import io, math, os, urllib.request, urllib.parse, json, shutil, subprocess
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
DATE = "2026-05-30"     # Saturday — recent date with full API coverage

# Same AOI as the bike script: 4 x 4 km around Helsinki Central Station
# in EPSG:3879 (Helsinki's cadastral CRS).
MINX, MINY = 25494800, 6671200
MAXX, MAXY = 25498800, 6675200
PX = 0.5
MAX_PX = 6000
PX = max(PX, (MAXX - MINX) / MAX_PX, (MAXY - MINY) / MAX_PX)
W = int(round((MAXX - MINX) / PX)); H = int(round((MAXY - MINY) / PX))
transform = from_origin(MINX, MAXY, PX, PX)
print(f"AOI {MAXX-MINX:.0f} x {MAXY-MINY:.0f} m -> {W} x {H} px @ {PX:.3f} m/px")

WFS = "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"
LIDOTIKU = "https://lidotiku.api.hel.fi/api"

CACHE = os.path.expanduser("~/.cache/helsinki_pedestrian")
os.makedirs(CACHE, exist_ok=True)

# --- 1) YLRE light-traffic polygons (sidewalks + ped/bike mixed surfaces) ---
ylre_cache = os.path.join(CACHE, "ylre_ped.geojson")
if not os.path.exists(ylre_cache):
    print("Fetching YLRE light-traffic polygons...")
    r = requests.get(WFS, params={
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": "avoindata:YLRE_Katu_ja_viherosat_kevytliikenne_alue",
        "srsName": "EPSG:3879",
        "bbox": f"{MINX},{MINY},{MAXX},{MAXY},EPSG:3879",
        "outputFormat": "application/json", "count": 50000}, timeout=600)
    r.raise_for_status()
    with open(ylre_cache, "wb") as f: f.write(r.content)
ped_polys = gpd.read_file(ylre_cache)
print(f"  YLRE light-traffic polygons: {len(ped_polys)}")

geoms = [g for g in ped_polys.geometry if g is not None and not g.is_empty]
ped_mask = rasterize([(g, 1) for g in geoms], out_shape=(H, W),
                     transform=transform, fill=0, dtype="uint8").astype(bool)
# Blue base for the pedestrian surface; visually distinct from the bike script.
base = np.full((H, W, 3), 255, dtype="uint8")
base[ped_mask] = (30, 90, 200)

# --- 2) Pedestrian Eco-Counter locations -----------------------------------
print("Fetching EcoCounter counter list...")
url = f"{LIDOTIKU}/counters/?format=json&source=EcoCounter&page_size=100"
with urllib.request.urlopen(url, timeout=60) as r:
    feats = json.load(r)["results"]["features"]
print(f"  EcoCounter stations total: {len(feats)}")

# Coords are EPSG:4326; project to EPSG:3879 to match the base raster.
to_local = Transformer.from_crs("EPSG:4326", "EPSG:3879", always_xy=True)

# Pre-fetch which counters have pedestrian observations. We only keep
# Eco-Counters whose first page of hourly observations contains at least
# one `pedestrian` row — cheap probe that avoids hammering the API later.
ped_sites = []
for f in feats:
    cid = f["properties"]["id"]
    name = f["properties"]["name"]
    lon, lat = f["geometry"]["coordinates"]
    if lon == 0 and lat == 0:
        continue
    x, y = to_local.transform(lon, lat)
    px = (x - MINX) / (MAXX - MINX) * W
    py = (MAXY - y) / (MAXY - MINY) * H
    if not (0 <= px < W and 0 <= py < H):
        continue
    probe = f"{LIDOTIKU}/observations/?format=json&counter={cid}&period=hour&page_size=5"
    try:
        with urllib.request.urlopen(probe, timeout=30) as r:
            obs = json.load(r)["results"]
        if any(o["vehicletype"] == "pedestrian" for o in obs):
            ped_sites.append({"id": cid, "name": name, "px": px, "py": py})
    except Exception as e:
        print(f"    probe failed for {cid}: {e}")
print(f"  pedestrian counters inside AOI: {len(ped_sites)}")
if not ped_sites:
    raise SystemExit("No pedestrian counters found in AOI — widen MINX..MAXY or change DATE.")

# --- 3) Hourly pedestrian counts for DATE ----------------------------------
# Lidotiku returns observations in reverse-chronological pages of ~100.
# We paginate per counter until we either pass DATE or run out of results.
end_iso = f"{DATE}T23:59:59+03:00"
start_iso = f"{DATE}T00:00:00+03:00"

hour_to_counts: dict[int, dict[str, int]] = {h: {} for h in range(24)}
for s in ped_sites:
    cid = s["id"]
    cache_f = os.path.join(CACHE, f"obs_{cid}_{DATE}.json")
    if os.path.exists(cache_f):
        with open(cache_f) as fh: rows = json.load(fh)
    else:
        rows = []
        next_url = (f"{LIDOTIKU}/observations/?format=json&counter={cid}"
                    f"&period=hour&start={DATE}&end={DATE}")
        while next_url:
            with urllib.request.urlopen(next_url, timeout=60) as r:
                payload = json.load(r)
            for o in payload["results"]:
                if o["vehicletype"] != "pedestrian": continue
                if start_iso <= o["datetime"] <= end_iso:
                    rows.append({"datetime": o["datetime"],
                                 "value": o.get("value") or 0})
            # Stop paginating once the page's oldest row is earlier than DATE.
            if payload["results"] and payload["results"][-1]["datetime"] < start_iso:
                break
            next_url = payload.get("next")
        with open(cache_f, "w") as fh: json.dump(rows, fh)
    for row in rows:
        h = int(row["datetime"][11:13])
        hour_to_counts[h][s["name"]] = hour_to_counts[h].get(s["name"], 0) + int(row["value"])
total_obs = sum(len(v) for v in hour_to_counts.values())
print(f"  pulled {total_obs} hourly pedestrian observations across "
      f"{len(ped_sites)} counters")

# --- 4) Render frames -------------------------------------------------------
SCALE = 0.20
small = np.array(Image.fromarray(base).resize(
    (int(W * SCALE), int(H * SCALE)), Image.LANCZOS))
sH, sW, _ = small.shape

maxc = max((max(v.values()) if v else 0) for v in hour_to_counts.values()) or 1
print(f"  peak single-station hourly count: {maxc}")

frame_dir = os.path.join(CACHE, f"frames_{DATE}")
os.makedirs(frame_dir, exist_ok=True)
for fname in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, fname))

for h in range(24):
    vals = hour_to_counts.get(h, {})
    fig, ax = plt.subplots(figsize=(sW / 100, sH / 100), dpi=100)
    ax.imshow(small); ax.set_xlim(0, sW); ax.set_ylim(sH, 0); ax.axis("off")
    for s in ped_sites:
        n = vals.get(s["name"], 0)
        if n <= 0: continue
        r = 4 + 30 * math.sqrt(n / maxc)
        ax.add_patch(mpatches.Circle(
            (s["px"] * SCALE, s["py"] * SCALE), radius=r,
            facecolor=(1.0, 0.65, 0.1, 0.75),
            edgecolor=(0.6, 0.35, 0.0, 0.95), linewidth=1.2))
    ax.text(10, 30, f"Helsinki pedestrian counters — {DATE}  {h:02d}:00",
            color="white", fontsize=16, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    total = sum(vals.values())
    ax.text(10, sH - 15,
            f"{total:,} pedestrians past {len(ped_sites)} sensors this hour",
            color="white", fontsize=12,
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    fig.savefig(f"{frame_dir}/{h:02d}.png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
print(f"  rendered 24 frames -> {frame_dir}")

# --- 5) Encode MP4 + GIF ---------------------------------------------------
out_mp4 = f"helsinki_pedestrian_{DATE}.mp4"
out_gif = f"helsinki_pedestrian_{DATE}.gif"
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
