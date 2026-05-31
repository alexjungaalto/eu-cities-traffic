"""Self-contained animation: Madrid traffic counters over one day.

Each ARROW is the *net flow* between a pair of opposite-direction sensors
at the same physical intersection. Arrow length and width both scale
with |net|. So a stretch of road shows a thick arrow inbound during the
morning rush and outbound in the evening, even where both carriageways
carry comparable traffic.

Data sources, all anonymous:
  * Sensor locations (~4000 inductive loops):
        https://datos.madrid.es/dataset/202468-0-intensidad-trafico/
        resource/202468-150-intensidad-trafico-csv/download/...csv
  * Historical 15-minute counts (one month per ZIP, ~80 MB compressed):
        https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/
        resource/208627-{N}-transporte-ptomedida-historico-zip/download/...zip
    (The integer {N} is NOT chronological — March 2024 = 60.)
  * Road network for the backdrop: OpenStreetMap via Overpass.

Output: madrid_<DATE>.mp4 and .gif.
"""
import io, math, os, csv, json, re, shutil, subprocess, urllib.request, zipfile
from datetime import date
from collections import defaultdict
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
DATE = "2024-03-13"
# March 2024 zip resource code (NOT chronological; see Madrid downloads page).
MONTH_RES = 60
MONTH_ZIP_URL = (f"https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/"
                 f"resource/208627-{MONTH_RES}-transporte-ptomedida-historico-zip/"
                 f"download/208627-{MONTH_RES}-transporte-ptomedida-historico-zip.zip")
LOCS_URL = ("https://datos.madrid.es/dataset/202468-0-intensidad-trafico/"
            "resource/202468-150-intensidad-trafico-csv/download/"
            "202468-150-intensidad-trafico-csv.csv")

# AOI: ~10 x 8 km centred on Puerta del Sol.
LAT_MIN, LAT_MAX = 40.380, 40.460
LON_MIN, LON_MAX = -3.760, -3.650

_aoi = gpd.GeoSeries.from_xy(
    [LON_MIN, LON_MAX, LON_MIN, LON_MAX],
    [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX], crs="EPSG:4326").to_crs(32630)
MINX, MAXX = float(_aoi.x.min()), float(_aoi.x.max())
MINY, MAXY = float(_aoi.y.min()), float(_aoi.y.max())

PX = 1.0
MAX_PX = 6000
PX = max(PX, (MAXX-MINX)/MAX_PX, (MAXY-MINY)/MAX_PX)
W = int(round((MAXX-MINX)/PX)); H = int(round((MAXY-MINY)/PX))
transform = from_origin(MINX, MAXY, PX, PX)
print(f"AOI {MAXX-MINX:.0f} x {MAXY-MINY:.0f} m -> {W} x {H} px @ {PX:.2f} m/px")

CACHE = os.path.expanduser("~/.cache/madrid_traffic")
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
}

def overpass(query):
    last = None
    for url in OVERPASS_MIRRORS:
        try:
            r = requests.post(url, data={"data": query},
                              headers={"User-Agent": "kbw-madrid/1.0"}, timeout=300)
            r.raise_for_status(); return r.json()
        except Exception as e:
            print(f"  overpass mirror {url} failed: {e}"); last = e
    raise last

# --- 1) Road backdrop -------------------------------------------------------
roads_cache = os.path.join(CACHE, "roads.geojson")
if not os.path.exists(roads_cache):
    print("Fetching OSM major roads...")
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
        feats.append({"type":"Feature",
                      "geometry":{"type":"LineString","coordinates":coords},
                      "properties":{"highway":hwy}})
    with open(roads_cache, "w") as f:
        json.dump({"type":"FeatureCollection","features":feats}, f)
roads = gpd.read_file(roads_cache).to_crs(32630)
print(f"  road centerlines: {len(roads)}")
roads_buf = roads.copy()
roads_buf["geometry"] = roads_buf.apply(
    lambda r: r.geometry.buffer(ROAD_WIDTH[r["highway"]] / 2), axis=1)
geoms = [g for g in roads_buf.geometry if g is not None and not g.is_empty]
car_mask = rasterize([(g, 1) for g in geoms], out_shape=(H, W),
                     transform=transform, fill=0, dtype="uint8").astype(bool)
base = np.full((H, W, 3), 255, dtype="uint8")
base[car_mask] = (220, 30, 30)

# --- 2) Sensor locations ----------------------------------------------------
locs_path = os.path.join(CACHE, "locations.csv")
if not os.path.exists(locs_path):
    print("Fetching sensor locations CSV...")
    urllib.request.urlretrieve(LOCS_URL, locs_path)
locs = pd.read_csv(locs_path, sep=";", encoding="latin-1", on_bad_lines="skip")
locs.columns = [c.strip().lower() for c in locs.columns]
print(f"  sensor locations total: {len(locs)}")
# Reproject UTM zone 30 -> our metric grid (same CRS already, but safe).
# UTM_X/UTM_Y may be in EPSG:25830 ETRS89 -- treat them as 25830.
glocs = gpd.GeoDataFrame(locs.copy(),
                          geometry=gpd.points_from_xy(locs["utm_x"], locs["utm_y"]),
                          crs="EPSG:25830").to_crs(32630)
glocs["xm"], glocs["ym"] = glocs.geometry.x, glocs.geometry.y
in_aoi = glocs[(glocs.xm >= MINX)&(glocs.xm < MAXX)&
               (glocs.ym >= MINY)&(glocs.ym < MAXY)].copy()
in_aoi["px"] = (in_aoi.xm - MINX) / (MAXX - MINX) * W
in_aoi["py"] = (MAXY - in_aoi.ym) / (MAXY - MINY) * H
print(f"  sensors inside AOI: {len(in_aoi)} of {len(glocs)}")

# --- 2b) Parse direction tokens + base direction (cardinal -> unit vector) -
# Sensor names look like
#   "Jose Ortega y Gasset E-O - Pº Castellana-Serrano"  (east->west)
#   "Jose Ortega y Gasset O-E - Serrano-Pº Castellana"  (west->east)
# We extract the token (E-O, O-E, N-S, S-N, NE-SO, SO-NE, NO-SE, SE-NO),
# turn it into a unit vector (image-coords with y flipped for plotting),
# and use the rest of the name as a "location key" so we can pair the
# two sensors at the same intersection.
DIR_VECTORS = {
    # (in image coords: +x = right/east, +y = down/south)
    "O-E": ( 1.0,  0.0),  # west -> east
    "E-O": (-1.0,  0.0),  # east -> west
    "N-S": ( 0.0,  1.0),  # north -> south
    "S-N": ( 0.0, -1.0),  # south -> north
    "NO-SE": ( 1.0,  1.0),
    "SE-NO": (-1.0, -1.0),
    "NE-SO": (-1.0,  1.0),
    "SO-NE": ( 1.0, -1.0),
}
def _norm(u, v):
    n = math.hypot(u, v)
    return (u/n, v/n) if n > 0 else (1.0, 0.0)
DIR_VECTORS = {k: _norm(*v) for k, v in DIR_VECTORS.items()}
TOK_RE = re.compile(r"\b(O-E|E-O|N-S|S-N|NO-SE|SE-NO|NE-SO|SO-NE)\b")

def parse(name):
    if not isinstance(name, str): return None, None
    m = TOK_RE.search(name)
    if not m: return None, None
    tok = m.group(1)
    # Location key = name minus the token, normalised.
    loc = TOK_RE.sub("", name).replace("  ", " ").strip(" -")
    return tok, loc.lower()

in_aoi[["dir_tok", "loc_key"]] = in_aoi["nombre"].apply(
    lambda n: pd.Series(parse(n)))
parsed = in_aoi.dropna(subset=["dir_tok"]).copy()
print(f"  sensors with parseable direction: {len(parsed)}")

# --- 2c) Pair sensors purely by spatial proximity --------------------------
# For each sensor, find the nearest sensor whose direction token is the
# reverse, within 80 m. Names are too inconsistent between paired
# sensors (cross-street order flips) to use as a matching key.
OPP = {"O-E":"E-O", "E-O":"O-E", "N-S":"S-N", "S-N":"N-S",
       "NO-SE":"SE-NO", "SE-NO":"NO-SE", "NE-SO":"SO-NE", "SO-NE":"NE-SO"}
MAX_PAIR_DIST_M = 80
pairs = []
seen = set()
# Build a per-direction index so the inner loop is O(n_per_dir) per sensor.
by_dir = {tok: parsed[parsed["dir_tok"] == tok].reset_index(drop=True)
          for tok in OPP}
for _, a in parsed.iterrows():
    aid = int(a["id"])
    if aid in seen: continue
    cand = by_dir.get(OPP[a["dir_tok"]])
    if cand is None or len(cand) == 0: continue
    d = ((cand["xm"] - a["xm"])**2 + (cand["ym"] - a["ym"])**2) ** 0.5
    j = d.idxmin()
    if d.loc[j] > MAX_PAIR_DIST_M: continue
    bid = int(cand.loc[j, "id"])
    if bid in seen or bid == aid: continue
    midx = (a["xm"] + cand.loc[j, "xm"]) / 2
    midy = (a["ym"] + cand.loc[j, "ym"]) / 2
    px = (midx - MINX) / (MAXX - MINX) * W
    py = (MAXY - midy) / (MAXY - MINY) * H
    pairs.append({"a": aid, "b": bid, "vec": DIR_VECTORS[a["dir_tok"]],
                  "px": px, "py": py})
    seen.add(aid); seen.add(bid)
print(f"  paired sensors: {len(pairs)*2} forming {len(pairs)} arrows")

# --- 3) Per-pair hourly net flow on DATE ------------------------------------
print(f"Fetching historical zip for month (resource {MONTH_RES})...")
zip_path = os.path.join(CACHE, "historico.zip")
if not os.path.exists(zip_path):
    urllib.request.urlretrieve(MONTH_ZIP_URL, zip_path)
with zipfile.ZipFile(zip_path) as z:
    csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
    csv_path = os.path.join(CACHE, csv_name)
    if not os.path.exists(csv_path):
        print(f"  extracting {csv_name} ...")
        z.extract(csv_name, CACHE)

ids_a = {p["a"] for p in pairs}
ids_b = {p["b"] for p in pairs}
ids_all = ids_a | ids_b

# Stream the 800 MB CSV; keep only rows on DATE whose id is in our pair set.
# We sum `intensidad` per (id, hour) — it's vehicles/h, so 15-min bins of the
# SAME hour just average; sum / 4 gives the hourly mean.
print("Streaming historical CSV (filter date + AOI ids)...")
per_id_hour = defaultdict(lambda: [0, 0])   # [sum_intensity, n_bins]
target_date = DATE
with open(csv_path, encoding="latin-1") as f:
    rdr = csv.reader(f, delimiter=";")
    header = next(rdr)
    for row in rdr:
        try:
            ts = row[1]
            if not ts.startswith(target_date): continue
            sid = int(row[0])
            if sid not in ids_all: continue
            if row[7] == "E": continue        # 'error' flag — skip faulty
            h = int(ts[11:13])
            intensity = int(row[3])
            acc = per_id_hour[(sid, h)]
            acc[0] += intensity; acc[1] += 1
        except Exception:
            continue
# Average per hour (intensity is vehicles/h, 15-min bins average to hourly).
mean_int = {(sid, h): acc[0]/acc[1] for (sid, h), acc in per_id_hour.items()
            if acc[1] > 0}
print(f"  hourly intensity values: {len(mean_int)}")

# --- 4) Render frames -------------------------------------------------------
SCALE = 0.18
small = np.array(Image.fromarray(base).resize(
    (int(W*SCALE), int(H*SCALE)), Image.LANCZOS))
sH, sW, _ = small.shape

# Peak |net| across the day.
maxnet = 1
for p in pairs:
    for h in range(24):
        a = mean_int.get((p["a"], h), 0)
        b = mean_int.get((p["b"], h), 0)
        maxnet = max(maxnet, abs(a - b))
print(f"  peak |net| (veh/h): {maxnet:.0f}")

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
    for p in pairs:
        a = mean_int.get((p["a"], h), 0)
        b = mean_int.get((p["b"], h), 0)
        net = a - b
        if net == 0: continue
        sign = 1 if net > 0 else -1
        dx, dy = p["vec"]                 # already in image coords
        L, w, hw, hl = arrow_dims(abs(net))
        x = p["px"] * SCALE; y = p["py"] * SCALE
        ax.add_patch(FancyArrow(
            x, y, sign*dx*L, sign*dy*L,
            width=w, head_width=hw, head_length=hl,
            length_includes_head=True,
            facecolor=(0.95, 0.40, 0.05, 0.90),
            edgecolor=(0.55, 0.15, 0.0, 0.95), linewidth=0.6))
    ax.text(10, 30, f"Madrid traffic loops — net flow per pair — {DATE}  {h:02d}:00",
            color="white", fontsize=13, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=4))
    fig.savefig(f"{frame_dir}/{h:02d}.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
print(f"  rendered 24 frames -> {frame_dir}")

# --- 5) Encode MP4 + GIF ---------------------------------------------------
out_mp4 = f"madrid_{DATE}.mp4"
out_gif = f"madrid_{DATE}.gif"
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
