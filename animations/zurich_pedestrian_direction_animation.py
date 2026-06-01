"""Self-contained animation: Zürich pedestrian-counter hourly flow with
DIRECTIONAL split (FUSS_IN / FUSS_OUT) over one day.

Why Zürich: the city's Tiefbauamt publishes 15-minute counts for both
pedestrians AND cyclists at every automated station, *with* an
explicit IN/OUT split (`FUSS_IN`, `FUSS_OUT`). That gives us
directional info Helsinki's pedestrian feed does not.

Data sources
------------
* Counter values: data.stadt-zuerich.ch dataset
  `ted_taz_verkehrszaehlungen_werte_fussgaenger_velo` — yearly CSVs,
  schema: FK_STANDORT, DATUM (15-min), VELO_IN, VELO_OUT, FUSS_IN,
  FUSS_OUT, OST, NORD (EPSG:2056 — Swiss LV95).
* Water bodies for context: OSM Overpass (Limmat + Zürichsee outlines).

Visual encoding
---------------
At each station we draw TWO stacked arrows:
  * orange, pointing right — FUSS_IN  this hour
  * blue,   pointing left  — FUSS_OUT this hour
Arrow length ∝ √count. The IN/OUT axis is artistic (not real
compass) because the CSV does not export per-station bearing.

Output: zurich_pedestrian_<DATE>.mp4 and .gif.
"""
import io, math, os, urllib.request, urllib.parse, json, shutil, subprocess
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from PIL import Image

# --- Config -----------------------------------------------------------------
DATE = "2026-05-28"      # Thursday — typical commuter day
YEAR = DATE[:4]

# AOI in EPSG:2056 (LV95). Wide 12 x 10 km box covering most of the
# city of Zürich — many pedestrian/bike counters sit on bridges and
# path entries outside the narrow Altstadt, so a small AOI misses them.
MINX, MINY = 2677000, 1241000
MAXX, MAXY = 2689000, 1251000
PX = 2.0
W = int((MAXX - MINX) / PX); H = int((MAXY - MINY) / PX)
transform = from_origin(MINX, MAXY, PX, PX)
print(f"AOI {MAXX-MINX:.0f} x {MAXY-MINY:.0f} m -> {W} x {H} px @ {PX} m/px")

VALUES_URL = (f"https://data.stadt-zuerich.ch/dataset/"
              f"ted_taz_verkehrszaehlungen_werte_fussgaenger_velo/download/"
              f"{YEAR}_verkehrszaehlungen_werte_fussgaenger_velo.csv")

CACHE = os.path.expanduser("~/.cache/zurich_pedestrian")
os.makedirs(CACHE, exist_ok=True)

# --- 1) Water bodies (Limmat + Zürichsee) via Overpass for visual context ---
water_cache = os.path.join(CACHE, "water.geojson")
if not os.path.exists(water_cache):
    print("Fetching water bodies via Overpass...")
    # Overpass uses WGS84; convert AOI bbox.
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon_min, lat_min = t.transform(MINX, MINY)
    lon_max, lat_max = t.transform(MAXX, MAXY)
    query = f"""
[out:json][timeout:60];
(
  way["natural"="water"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["waterway"]({lat_min},{lon_min},{lat_max},{lon_max});
  relation["natural"="water"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out geom;
"""
    r = requests.post("https://overpass-api.de/api/interpreter",
                      data={"data": query},
                      headers={"User-Agent": "cai2vo-demo/1.0"},
                      timeout=120)
    r.raise_for_status()
    elems = r.json()["elements"]
    feats = []
    for e in elems:
        if e["type"] == "way" and "geometry" in e:
            coords = [[p["lon"], p["lat"]] for p in e["geometry"]]
            if coords[0] == coords[-1] and len(coords) >= 4:
                feats.append({"type": "Feature", "properties": {},
                              "geometry": {"type": "Polygon",
                                           "coordinates": [coords]}})
            else:
                feats.append({"type": "Feature", "properties": {},
                              "geometry": {"type": "LineString",
                                           "coordinates": coords}})
    with open(water_cache, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)

water = gpd.read_file(water_cache).set_crs(4326).to_crs(2056)
print(f"  water features: {len(water)}")

# Render base: light gray canvas, water in pale blue, river lines thickened.
base = np.full((H, W, 3), 245, dtype="uint8")
poly_geoms = [g for g in water.geometry
              if g is not None and not g.is_empty
              and g.geom_type in ("Polygon", "MultiPolygon")]
if poly_geoms:
    water_mask = rasterize([(g, 1) for g in poly_geoms], out_shape=(H, W),
                           transform=transform, fill=0, dtype="uint8").astype(bool)
    base[water_mask] = (180, 215, 240)
line_geoms = [g.buffer(6.0) for g in water.geometry
              if g is not None and not g.is_empty
              and g.geom_type in ("LineString", "MultiLineString")]
if line_geoms:
    line_mask = rasterize([(g, 1) for g in line_geoms], out_shape=(H, W),
                          transform=transform, fill=0, dtype="uint8").astype(bool)
    base[line_mask] = (180, 215, 240)

# --- 2) Counter values for DATE --------------------------------------------
csv_cache = os.path.join(CACHE, f"values_{YEAR}.csv")
if not os.path.exists(csv_cache):
    print(f"Fetching {YEAR} values CSV (~14 MB)...")
    urllib.request.urlretrieve(VALUES_URL, csv_cache)

print("Reading CSV...")
df = pd.read_csv(csv_cache, parse_dates=["DATUM"])
print(f"  total rows: {len(df):,}")
day = df[df["DATUM"].dt.strftime("%Y-%m-%d") == DATE].copy()
day["hour"] = day["DATUM"].dt.hour
# Only stations with pedestrian sensors and coordinates inside the AOI.
day = day.dropna(subset=["OST", "NORD"])
day = day[(day["OST"].between(MINX, MAXX)) & (day["NORD"].between(MINY, MAXY))]
print(f"  rows on {DATE} inside AOI: {len(day)}")

# Aggregate 15-min slots into hourly counts per station.
agg = (day.groupby(["FK_STANDORT", "hour"])
          .agg(fuss_in=("FUSS_IN", "sum"),
               fuss_out=("FUSS_OUT", "sum"),
               ost=("OST", "first"),
               nord=("NORD", "first"))
          .reset_index())
agg = agg[(agg["fuss_in"] + agg["fuss_out"]) > 0]
sites = agg[["FK_STANDORT", "ost", "nord"]].drop_duplicates()
print(f"  pedestrian-active stations inside AOI: {len(sites)}")

sites["px"] = (sites["ost"] - MINX) / (MAXX - MINX) * W
sites["py"] = (MAXY - sites["nord"]) / (MAXY - MINY) * H

hour_data: dict[int, list[dict]] = {h: [] for h in range(24)}
for _, r in agg.iterrows():
    s = sites[sites["FK_STANDORT"] == r["FK_STANDORT"]].iloc[0]
    hour_data[int(r["hour"])].append({
        "px": s["px"], "py": s["py"],
        "in": int(r["fuss_in"]), "out": int(r["fuss_out"]),
    })

# --- 3) Render frames -------------------------------------------------------
SCALE = 0.40                  # bump up because AOI is smaller than Helsinki
small = np.array(Image.fromarray(base).resize(
    (int(W * SCALE), int(H * SCALE)), Image.LANCZOS))
sH, sW, _ = small.shape

# Peak single-direction value for consistent arrow scaling across the day.
maxc = max((max((d["in"] for d in lst), default=0) for lst in hour_data.values()),
           default=1)
maxc = max(maxc, max((max((d["out"] for d in lst), default=0)
                      for lst in hour_data.values()), default=1), 1)
print(f"  peak single-direction hourly count: {maxc}")

ARROW_MAX_PX = 36
ARROW_MIN_PX = 5
def arrow_len(n: int) -> float:
    if n <= 0: return 0
    return ARROW_MIN_PX + (ARROW_MAX_PX - ARROW_MIN_PX) * math.sqrt(n / maxc)

frame_dir = os.path.join(CACHE, f"frames_{DATE}")
os.makedirs(frame_dir, exist_ok=True)
for fname in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, fname))

for h in range(24):
    fig, ax = plt.subplots(figsize=(sW / 100, sH / 100), dpi=100)
    ax.imshow(small); ax.set_xlim(0, sW); ax.set_ylim(sH, 0); ax.axis("off")
    for d in hour_data[h]:
        x = d["px"] * SCALE; y = d["py"] * SCALE
        # IN arrow above, pointing right (orange).
        Li = arrow_len(d["in"])
        if Li > 0:
            ax.annotate("", xy=(x + Li, y - 3), xytext=(x, y - 3),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=(1.0, 0.55, 0.0),
                                        lw=1.4, alpha=0.9,
                                        mutation_scale=10))
        # OUT arrow below, pointing left (blue).
        Lo = arrow_len(d["out"])
        if Lo > 0:
            ax.annotate("", xy=(x - Lo, y + 3), xytext=(x, y + 3),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=(0.10, 0.40, 0.85),
                                        lw=1.4, alpha=0.9,
                                        mutation_scale=10))
    ax.text(10, 28, f"Zürich pedestrian flow — {DATE}  {h:02d}:00",
            color="white", fontsize=15, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    tot_in = sum(d["in"]  for d in hour_data[h])
    tot_out = sum(d["out"] for d in hour_data[h])
    ax.text(10, sH - 15,
            f"IN {tot_in:,}  /  OUT {tot_out:,}  "
            f"across {len(hour_data[h])} active stations",
            color="white", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    # Mini legend
    ax.annotate("", xy=(sW - 60, 28), xytext=(sW - 90, 28),
                arrowprops=dict(arrowstyle="-|>", color=(1.0, 0.55, 0.0),
                                lw=1.4, mutation_scale=10))
    ax.text(sW - 55, 30, "IN", color="white", fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6, pad=2))
    ax.annotate("", xy=(sW - 90, 46), xytext=(sW - 60, 46),
                arrowprops=dict(arrowstyle="-|>", color=(0.10, 0.40, 0.85),
                                lw=1.4, mutation_scale=10))
    ax.text(sW - 55, 48, "OUT", color="white", fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6, pad=2))

    fig.savefig(f"{frame_dir}/{h:02d}.png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
print(f"  rendered 24 frames -> {frame_dir}")

# --- 4) Encode MP4 + GIF ---------------------------------------------------
out_mp4 = f"zurich_pedestrian_{DATE}.mp4"
out_gif = f"zurich_pedestrian_{DATE}.gif"
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
