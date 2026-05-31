"""Pixel-wise segmentation of central Milano using DBT 2012 polygons.

DBT (Database Topografico) is the Italian equivalent of YLRE/FMZK: a
surveyed, decimetre-accurate vector basemap. The Comune di Milano
publishes its 2012 DBT openly via an ArcGIS REST FeatureServer (no
login required). Three polygon classes are used here:

  A010101_area_circolazione_veicolare   -> motor carriageway
  A010102_area_circolazione_pedonale    -> sidewalk / pedestrian area
  A010103_area_circolazione_ciclabile   -> bike-lane polygon

Unlike Vienna (which only has bike LineStrings), Milano gives us true
bike-lane polygons, so this is an apples-to-apples comparison with
Helsinki's YLRE.

A caveat: the DBT class A010101 is "any surface motor vehicles may
drive on" — which includes pedestrianised squares open to delivery,
service, and emergency access (Piazza del Duomo, Largo Cairoli, Arco
della Pace, parts of Parco Sempione). Daily lived public space there
is pedestrian even though DBT calls it `circolazione_veicolare`.

To correct for this, we additionally fetch OSM polygons tagged
`highway=pedestrian` or `area:highway=pedestrian/footway` and use them
to reclassify the underlying DBT car pixels as pedestrian. The fix is
imperfect but matches the ground truth in well-tagged OSM areas.

Coordinate reference: EPSG:25832 (ETRS89 / UTM Zone 32N). Orthophoto
background comes from the ESRI World Imagery REST service.
"""
import io, math, numpy as np, requests, geopandas as gpd
from shapely.geometry import shape
from rasterio.transform import from_origin
from rasterio.features import rasterize
from PIL import Image

# --- Area Of Interest -------------------------------------------------------
# 4 x 4 km square centred on Piazza del Duomo (~514800, 5034630).
# Covers the inner ring (Cerchia dei Bastioni) and a margin around it.
minx, miny = 512800, 5032600   # SW corner (E, N) in metres
maxx, maxy = 516800, 5036600   # NE corner (E, N) in metres

# PX = ground resolution in metres/pixel. See helsinki.py for the rationale.
PX = 0.5
MAX_PX = 6000
PX = max(PX, (maxx - minx) / MAX_PX, (maxy - miny) / MAX_PX)
W = int(round((maxx - minx) / PX)); H = int(round((maxy - miny) / PX))
transform = from_origin(minx, maxy, PX, PX)
print(f"AOI {(maxx-minx):.0f} x {(maxy-miny):.0f} m  ->  {W} x {H} px @ {PX:.3f} m/px")

# --- Endpoints --------------------------------------------------------------
# Comune di Milano DBT 2012 via ArcGIS FeatureServer.
FS = ("https://geoportale.comune.milano.it/arcgis/rest/services/"
      "Cartografie_Vettoriali/DBT_2012_Milano_RDN2008_UTM32/FeatureServer")
# Layer IDs (probed via FeatureServer?f=json — the same names recur across
# scale tiers; these IDs are the tier with full coverage in the city core).
LAYER_IDS = {"car": 31, "ped": 23, "bike": 30}
# ESRI World Imagery — public REST service, takes a bbox in any CRS.
ORTHO = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"


def ortho_image(tile_px=2048):
    """Mosaic the ESRI World Imagery export into one (H, W, 3) RGB array.
    Tiles to stay under the per-request pixel cap."""
    out = np.zeros((H, W, 3), "uint8")
    nx = math.ceil(W / tile_px); ny = math.ceil(H / tile_px)
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * tile_px; y0 = iy * tile_px
            tw = min(tile_px, W - x0); th = min(tile_px, H - y0)
            tminx = minx + x0 * PX;   tmaxx = tminx + tw * PX
            tmaxy = maxy - y0 * PX;   tminy = tmaxy - th * PX
            r = requests.get(ORTHO, params={
                "bbox": f"{tminx},{tminy},{tmaxx},{tmaxy}",
                "bboxSR": "25832", "imageSR": "25832",
                "size": f"{tw},{th}", "format": "png", "f": "image"},
                timeout=600)
            r.raise_for_status()
            out[y0:y0+th, x0:x0+tw] = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
            print(f"  ortho tile {iy*nx+ix+1}/{nx*ny}")
    return out


def _query(layer_id, bb):
    """One GeoJSON query for the given (xmin,ymin,xmax,ymax) bbox."""
    r = requests.get(f"{FS}/{layer_id}/query", params={
        "where": "1=1",
        "geometry": f"{bb[0]},{bb[1]},{bb[2]},{bb[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "25832", "outSR": "25832",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*", "f": "geojson",
    }, timeout=600)
    r.raise_for_status()
    return r.json()


def fetch_layer(layer_id):
    """ArcGIS caps a single query at 2000 features and rejects resultOffset
    on this service, so we recursively subdivide the AOI into quadrants
    whenever a query saturates the cap. This is the standard ESRI-FS
    workaround when offset pagination is disabled."""
    feats = []
    stack = [(minx, miny, maxx, maxy)]
    while stack:
        bb = stack.pop()
        gj = _query(layer_id, bb)
        chunk = gj.get("features", [])
        exceeded = gj.get("exceededTransferLimit") or \
                   gj.get("properties", {}).get("exceededTransferLimit")
        # Stop recursing once the tile is small enough or returns < cap.
        if exceeded and (bb[2] - bb[0]) > 50:
            mx = (bb[0] + bb[2]) / 2; my = (bb[1] + bb[3]) / 2
            stack += [(bb[0], bb[1], mx, my), (mx, bb[1], bb[2], my),
                      (bb[0], my, mx, bb[3]), (mx, my, bb[2], bb[3])]
        else:
            feats.extend(chunk)
            if exceeded:
                print(f"    WARNING: still capped at 50 m tile, dropped features")
    # Deduplicate by feature id (recursive tiles can re-include polygons
    # that straddle a quadrant boundary).
    seen = set(); uniq = []
    for f in feats:
        fid = f.get("id") or f.get("properties", {}).get("OBJECTID")
        if fid in seen: continue
        seen.add(fid); uniq.append(f)
    return gpd.GeoDataFrame.from_features(uniq, crs="EPSG:25832")


print("Fetching orthophoto (ESRI World Imagery)...")
ortho = ortho_image()

gdfs = {}
for cls, lid in LAYER_IDS.items():
    print(f"Fetching {cls} (FeatureServer/{lid})...")
    g = fetch_layer(lid)
    print(f"  {cls}: {len(g)} polygons")
    gdfs[cls] = g


def rast(geoms):
    """Burn polygons onto the (H, W) raster -> uint8 mask (1 = covered)."""
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=transform,
                     fill=0, dtype="uint8") if geoms else np.zeros((H, W), "uint8")

# --- OSM pedestrianisation override -----------------------------------------
# Pull explicitly-tagged pedestrian *polygons* from OSM and use them to
# override the DBT car classification at places like Piazza del Duomo and
# Arco della Pace, which DBT calls vehicle-accessible because service
# traffic is allowed but which are pedestrian space in lived reality.
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
# AOI bbox needs to be in lat/lon for Overpass. Convert EPSG:25832 corners.
_corners = gpd.GeoSeries.from_xy(
    [minx, maxx, minx, maxx], [miny, miny, maxy, maxy], crs="EPSG:25832"
).to_crs(4326)
lat_min = float(_corners.y.min()); lat_max = float(_corners.y.max())
lon_min = float(_corners.x.min()); lon_max = float(_corners.x.max())
osm_q = f"""
[out:json][timeout:300];
(
  way["highway"="pedestrian"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["area:highway"~"pedestrian|footway"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["highway"~"pedestrian|footway"]["area"="yes"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out geom;
"""
ped_override = []
for url in OVERPASS_MIRRORS:
    try:
        r = requests.post(url, data={"data": osm_q},
                          headers={"User-Agent": "kbw-milano-demo/1.0"}, timeout=300)
        r.raise_for_status()
        osm = r.json()
        for el in osm.get("elements", []):
            if "geometry" not in el: continue
            coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(coords) < 3: continue
            if coords[0] != coords[-1]: coords.append(coords[0])
            ped_override.append({"type": "Feature",
                                  "geometry": {"type": "Polygon", "coordinates": [coords]},
                                  "properties": {}})
        break
    except Exception as e:
        print(f"  overpass mirror {url} failed: {e}")
print(f"OSM pedestrian-polygon override: {len(ped_override)} features")
if ped_override:
    ped_osm = gpd.GeoDataFrame.from_features(ped_override, crs="EPSG:4326").to_crs(25832)
    ped_osm_mask = rast(ped_osm.geometry).astype(bool)
else:
    ped_osm_mask = np.zeros((H, W), bool)

# Priority: bike > OSM-pedestrian-override > car > ped(DBT).
# OSM override beats car so pedestrianised squares stop showing as red.
bike_mask = rast(gdfs["bike"].geometry).astype(bool)
car_raw   = rast(gdfs["car"].geometry).astype(bool)
ped_raw   = rast(gdfs["ped"].geometry).astype(bool)
car_mask  = car_raw & ~ped_osm_mask & ~bike_mask
ped_mask  = (ped_raw | ped_osm_mask) & ~bike_mask & ~car_mask

palette = {"car": (220, 30, 30), "ped": (30, 90, 230), "bike": (255, 200, 0)}
seg = np.full_like(ortho, 255)
seg[car_mask]  = palette["car"]
seg[ped_mask]  = palette["ped"]
seg[bike_mask] = palette["bike"]

overlay = (0.45 * ortho + 0.55 * seg).astype("uint8")
Image.fromarray(seg).save("milano_dbt_labels.png")
Image.fromarray(overlay).save("milano_dbt_overlay.png")

px = lambda m: int(m.sum() * PX**2)
aoi = (maxx - minx) * (maxy - miny)
print(f"\n=== Milano DBT segmentation ({(maxx-minx):.0f}m x {(maxy-miny):.0f}m AOI) ===")
for name, m in [("Car carriageway", car_mask), ("Pedestrian", ped_mask),
                ("Bike lane", bike_mask)]:
    a = px(m); print(f"  {name:20s} {a:9,d} m²  ({a/aoi*100:5.2f}%)")
print("Saved milano_dbt_labels.png  milano_dbt_overlay.png")
