"""Pixel-wise segmentation of central Corniglia using OpenStreetMap.

Unlike the Helsinki/Vienna/Milano/Paris scripts (which use surveyed
cadastral surface polygons), this version falls back to OSM because no
free open-WFS surface dataset exists for Corniglia. The trade-off:

  + Works for any city in the world, no API key.
  - Roads are lines, not surface polygons. We buffer them by typical
    width per highway class to approximate carriageway area.
  - Sidewalks are usually lines too. Buffer 2 m per side.
  - Some pedestrian zones / piazzas are tagged as true polygons
    (`highway=pedestrian + area=yes`, `area:highway=*`); we use those.

Query goes through Overpass. AOI is in WGS84 for the query, then
reprojected to UTM 34N (EPSG:32632) for the metric raster grid.
Background imagery from ESRI World Imagery.
"""
import io, math, numpy as np, requests, geopandas as gpd
from shapely.geometry import shape
from rasterio.transform import from_origin
from rasterio.features import rasterize
from PIL import Image

# --- Area Of Interest -------------------------------------------------------
# 1.5 x 1.5 km centred on Corniglia. The only one of the five villages set inland on a cliff above the sea.
LAT0, LON0 = 44.1199, 9.7088
SIZE_M = 1500

DLAT = SIZE_M / 2 / 111_320
DLON = SIZE_M / 2 / (111_320 * math.cos(math.radians(LAT0)))
lat_min, lat_max = LAT0 - DLAT, LAT0 + DLAT
lon_min, lon_max = LON0 - DLON, LON0 + DLON

# Reproject AOI corners to UTM 34N for the metric grid.
_aoi = gpd.GeoSeries.from_xy(
    [lon_min, lon_max, lon_min, lon_max],
    [lat_min, lat_min, lat_max, lat_max], crs="EPSG:4326").to_crs(32632)
minx = float(_aoi.x.min()); maxx = float(_aoi.x.max())
miny = float(_aoi.y.min()); maxy = float(_aoi.y.max())

# PX = ground resolution in metres/pixel; see helsinki.py for rationale.
PX = 0.5
MAX_PX = 6000
PX = max(PX, (maxx - minx) / MAX_PX, (maxy - miny) / MAX_PX)
W = int(round((maxx - minx) / PX)); H = int(round((maxy - miny) / PX))
transform = from_origin(minx, maxy, PX, PX)
print(f"AOI {(maxx-minx):.0f} x {(maxy-miny):.0f} m  ->  {W} x {H} px @ {PX:.3f} m/px")

# Multiple Overpass mirrors — the main one (overpass-api.de) frequently
# returns 504. We try each in order until one responds.
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
ORTHO    = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"

# Typical effective widths per OSM highway class, used to buffer the
# centerlines into surface polygons. Numbers are conservative averages.
ROAD_WIDTH = {
    "motorway": 14, "trunk": 12, "primary": 10, "secondary": 8,
    "tertiary": 7, "unclassified": 6, "residential": 6,
    "living_street": 5, "service": 4, "track": 3,
    # 'pedestrian' is special — treated as ped, not car.
}


def overpass(query):
    last = None
    for url in OVERPASS_MIRRORS:
        try:
            r = requests.post(url, data={"data": query},
                              headers={"User-Agent": "kbw-corniglia-demo/1.0"}, timeout=300)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  overpass mirror {url} failed: {e}")
            last = e
    raise last


def osm_to_gdf(osm, want_geom_types):
    """Convert an Overpass result (with `out geom`) to a GeoDataFrame.
    `want_geom_types` is a set like {"LineString", "Polygon"} to filter."""
    rows = []
    for el in osm.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "way" and "geometry" in el:
            coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(coords) < 2: continue
            if tags.get("area") == "yes" or "area:highway" in tags:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                geom = shape({"type": "Polygon", "coordinates": [coords]})
                gt = "Polygon"
            else:
                geom = shape({"type": "LineString", "coordinates": coords})
                gt = "LineString"
            if gt not in want_geom_types: continue
            rows.append({"geometry": geom, **tags})
    if not rows:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(32632)


def rast(geoms):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=transform,
                     fill=0, dtype="uint8") if geoms else np.zeros((H, W), "uint8")


def ortho_image(tile_px=1024):
    """Retry-aware ESRI imagery mosaic. Smaller tiles than the other
    scripts because the global ArcGIS service occasionally 504s on big
    requests."""
    import time
    out = np.zeros((H, W, 3), "uint8")
    nx = math.ceil(W / tile_px); ny = math.ceil(H / tile_px)
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * tile_px; y0 = iy * tile_px
            tw = min(tile_px, W - x0); th = min(tile_px, H - y0)
            tminx = minx + x0 * PX;   tmaxx = tminx + tw * PX
            tmaxy = maxy - y0 * PX;   tminy = tmaxy - th * PX
            for attempt in range(4):
                try:
                    r = requests.get(ORTHO, params={
                        "bbox": f"{tminx},{tminy},{tmaxx},{tmaxy}",
                        "bboxSR": "32632", "imageSR": "32632",
                        "size": f"{tw},{th}", "format": "png", "f": "image"}, timeout=300)
                    r.raise_for_status()
                    out[y0:y0+th, x0:x0+tw] = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
                    break
                except Exception as e:
                    if attempt == 3: raise
                    time.sleep(2 * (attempt + 1))
            print(f"  ortho tile {iy*nx+ix+1}/{nx*ny}")
    return out


print("Fetching orthophoto (ESRI World Imagery)...")
ortho = ortho_image()

# One bulky Overpass query for everything road/path/cycle-related. We
# request `out geom` so each way carries inline coordinates.
bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
print("Fetching OSM ways via Overpass...")
osm = overpass(f"""
[out:json][timeout:300];
(
  way["highway"]({bbox});
  way["area:highway"]({bbox});
  way["footway"]({bbox});
  way["railway"]({bbox});
);
out geom;
""")
print(f"  elements: {len(osm.get('elements', []))}")

ways = osm_to_gdf(osm, {"LineString", "Polygon"})
ways["highway"] = ways.get("highway")
ways["area_hwy"] = ways.get("area:highway")
ways["railway"] = ways.get("railway")
print(f"  ways parsed: {len(ways)}")

# ---- Build masks -----------------------------------------------------------
# 1) True pedestrian polygons (well-tagged piazzas, pedestrianised streets).
ped_polys = ways[ways.geometry.geom_type == "Polygon"]
ped_polys = ped_polys[
    ped_polys["highway"].isin(["pedestrian", "footway", "living_street"]) |
    ped_polys["area_hwy"].isin(["pedestrian", "footway", "sidewalk", "living_street"])
]

# 2) Buffer pedestrian-only road LINES (footways, paths, pedestrian streets,
# living streets) by 2 m -> 4 m wide ribbon.
ped_lines = ways[
    (ways.geometry.geom_type == "LineString") &
    (ways["highway"].isin(["footway", "path", "pedestrian", "steps", "living_street"]))
]
ped_lines = ped_lines.assign(geometry=ped_lines.geometry.buffer(2.0))

# 3) Bike lines (highway=cycleway).
bike_lines = ways[
    (ways.geometry.geom_type == "LineString") &
    (ways["highway"] == "cycleway")
]
bike_lines = bike_lines.assign(geometry=bike_lines.geometry.buffer(1.25))

# 4) Carriageway lines: every motor road class. Width depends on class.
car_lines = ways[
    (ways.geometry.geom_type == "LineString") &
    (ways["highway"].isin(ROAD_WIDTH.keys()))
].copy()
car_lines["geometry"] = car_lines.apply(
    lambda r: r.geometry.buffer(ROAD_WIDTH[r["highway"]] / 2), axis=1)

print(f"  ped polys: {len(ped_polys)}  ped lines (buffered): {len(ped_lines)}")
print(f"  bike (buffered): {len(bike_lines)}  car (buffered): {len(car_lines)}")


# 5) Railway lines (mostly the Cinque Terre / Genova-La Spezia line through
# tunnels and along the coast). Buffer to ~2.5 m per side for a 5 m ribbon.
rail_lines = ways[
    (ways.geometry.geom_type == "LineString") &
    (ways["railway"].isin(["rail", "light_rail", "narrow_gauge", "subway"]))
]
rail_lines = rail_lines.assign(geometry=rail_lines.geometry.buffer(2.5))
print(f"  rail (buffered): {len(rail_lines)}")

car_mask  = rast(car_lines.geometry).astype(bool)
bike_mask = rast(bike_lines.geometry).astype(bool) & ~car_mask
ped_mask  = (rast(ped_polys.geometry).astype(bool) |
             rast(ped_lines.geometry).astype(bool)) & ~car_mask & ~bike_mask

rail_mask = rast(rail_lines.geometry).astype(bool)
palette = {"car": (220, 30, 30), "ped": (30, 90, 230),
           "bike": (255, 200, 0), "rail": (255, 105, 180)}
seg = np.full_like(ortho, 255)
seg[car_mask]  = palette["car"]
seg[ped_mask]  = palette["ped"]
seg[bike_mask] = palette["bike"]
seg[rail_mask] = palette["rail"]

overlay = (0.45 * ortho + 0.55 * seg).astype("uint8")
Image.fromarray(seg).save("corniglia_osm_labels.png")
Image.fromarray(overlay).save("corniglia_osm_overlay.png")

px = lambda m: int(m.sum() * PX**2)
aoi = (maxx - minx) * (maxy - miny)
print(f"\n=== Corniglia OSM segmentation ({(maxx-minx):.0f}m x {(maxy-miny):.0f}m AOI) ===")
for name, m in [("Car carriageway", car_mask), ("Pedestrian", ped_mask),
                ("Bike lane", bike_mask), ("Railway", rail_mask)]:
    a = px(m); print(f"  {name:20s} {a:9,d} m²  ({a/aoi*100:5.2f}%)")
print("(OSM fallback: numbers are buffered estimates, not surveyed surfaces.)")
print("Saved corniglia_osm_labels.png  corniglia_osm_overlay.png")
