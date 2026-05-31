"""Pixel-wise segmentation of central Paris using the Ville de Paris
'Plan de voirie' open data.

The Ville de Paris publishes a surveyed, surface-accurate vector plan of
the public right-of-way as a set of separate datasets on
opendata.paris.fr. We use:

  denominations-emprises-voies-actuelles      -> total street envelope
  plan-de-voirie-trottoirs-emprises           -> sidewalk polygons
  plan-de-voirie-pistes-cyclables-et-couloirs-de-bus -> bike + bus lanes
  plan-de-voirie-passages-pietons             -> pedestrian crossings
  plan-de-voirie-aires-mixtes-vehicules-et-pietons   -> shared zones

There is no explicit 'chaussée' (carriageway) layer, so we compute it as
the street envelope minus everything else.

Datasets are served in WGS84 (EPSG:4326). We reproject to RGF93 /
Lambert-93 (EPSG:2154) for the metric raster grid. Orthophoto: ESRI
World Imagery REST service.
"""
import io, math, numpy as np, requests, geopandas as gpd, pandas as pd
from shapely.geometry import shape
from rasterio.transform import from_origin
from rasterio.features import rasterize
from PIL import Image

# --- Area Of Interest -------------------------------------------------------
# 4 x 4 km square centred near Notre-Dame / Île de la Cité.
# WGS84 bbox is used only for the opendata.paris.fr query; rasterisation
# happens in Lambert-93 (EPSG:2154) for an honest metric grid.
LAT0, LON0 = 48.8530, 2.3488            # Notre-Dame
SIZE_M = 4000                           # 4 km on each side

# Approximate degree-per-metre at Paris latitude.
DLAT = SIZE_M / 2 / 111_320
DLON = SIZE_M / 2 / (111_320 * math.cos(math.radians(LAT0)))
lat_min, lat_max = LAT0 - DLAT, LAT0 + DLAT
lon_min, lon_max = LON0 - DLON, LON0 + DLON

# Reproject the AOI corners to Lambert-93 to pin the raster grid.
_aoi_wgs = gpd.GeoSeries.from_xy(
    [lon_min, lon_max, lon_min, lon_max],
    [lat_min, lat_min, lat_max, lat_max], crs="EPSG:4326").to_crs(2154)
minx = float(_aoi_wgs.x.min()); maxx = float(_aoi_wgs.x.max())
miny = float(_aoi_wgs.y.min()); maxy = float(_aoi_wgs.y.max())

# PX = ground resolution in metres/pixel. See helsinki.py for the rationale.
PX = 0.5
MAX_PX = 6000
PX = max(PX, (maxx - minx) / MAX_PX, (maxy - miny) / MAX_PX)
W = int(round((maxx - minx) / PX)); H = int(round((maxy - miny) / PX))
transform = from_origin(minx, maxy, PX, PX)
print(f"AOI {(maxx-minx):.0f} x {(maxy-miny):.0f} m  ->  {W} x {H} px @ {PX:.3f} m/px")

# Bulk GeoJSON export endpoint (no 10k-row offset cap, unlike /records).
EXPORT = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/{ds}/exports/geojson"
ORTHO  = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"


def fetch_dataset(dataset_id):
    """Download one dataset clipped to the WGS84 AOI as GeoJSON, then
    reproject to Lambert-93 (EPSG:2154) for the metric raster."""
    where = f"in_bbox(geo_shape,{lat_min},{lon_min},{lat_max},{lon_max})"
    r = requests.get(EXPORT.format(ds=dataset_id),
                     params={"where": where}, timeout=600)
    r.raise_for_status()
    gdf = gpd.read_file(io.BytesIO(r.content))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(2154)


def ortho_image(tile_px=2048):
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
                "bboxSR": "2154", "imageSR": "2154",
                "size": f"{tw},{th}", "format": "png", "f": "image"}, timeout=600)
            r.raise_for_status()
            out[y0:y0+th, x0:x0+tw] = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
            print(f"  ortho tile {iy*nx+ix+1}/{nx*ny}")
    return out


print("Fetching orthophoto (ESRI World Imagery)...")
ortho = ortho_image()

print("Fetching street envelope (denominations-emprises-voies-actuelles)...")
voies = fetch_dataset("denominations-emprises-voies-actuelles")
print(f"  {len(voies)} polygons")

print("Fetching trottoirs (plan-de-voirie-trottoirs-emprises)...")
trottoirs = fetch_dataset("plan-de-voirie-trottoirs-emprises")
print(f"  {len(trottoirs)} polygons")

print("Fetching pistes cyclables + couloirs bus...")
pistes = fetch_dataset("plan-de-voirie-pistes-cyclables-et-couloirs-de-bus")
# Bike-only: lib_classe contains 'Piste cyclable'. Shared bus+bike (e.g.
# 'Couloir mixte bus-vélo') is debatable; we include it as bike since it
# is a designated cycling space.
bike_classes = pistes["lib_classe"].fillna("").str.lower()
bike_mask_sel = bike_classes.str.contains("piste cyclable") | \
                bike_classes.str.contains("mixte bus-vélo") | \
                bike_classes.str.contains("mixte bus-velo")
pistes_bike = pistes[bike_mask_sel]
print(f"  pistes total {len(pistes)}, bike-relevant {len(pistes_bike)}")
print(f"  lib_classe counts: {pistes['lib_classe'].value_counts().head(8).to_dict()}")

print("Fetching passages piétons + aires mixtes...")
passages = fetch_dataset("plan-de-voirie-passages-pietons")
aires    = fetch_dataset("plan-de-voirie-aires-mixtes-vehicules-et-pietons")
print(f"  passages {len(passages)}, aires {len(aires)}")


def rast(geoms):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=transform,
                     fill=0, dtype="uint8") if geoms else np.zeros((H, W), "uint8")


# Build the masks. Carriageway = inside the public-street envelope, not
# claimed by a sidewalk, bike lane, crossing or shared zone.
voirie_mask = rast(voies.geometry).astype(bool)
ped_base    = rast(trottoirs.geometry).astype(bool)
bike_base   = rast(pistes_bike.geometry).astype(bool)
ped_mask    = (ped_base | rast(passages.geometry).astype(bool) |
               rast(aires.geometry).astype(bool))
# Bike claims its pixels before ped, so a bike lane on a sidewalk reads
# as bike not ped (matches the Vienna/Milano logic).
bike_mask = bike_base & ~rast(trottoirs.geometry).astype(bool)
ped_mask  = ped_mask & ~bike_mask
car_mask  = voirie_mask & ~ped_mask & ~bike_mask

palette = {"car": (220, 30, 30), "ped": (30, 90, 230), "bike": (255, 200, 0)}
seg = np.full_like(ortho, 255)
seg[car_mask]  = palette["car"]
seg[ped_mask]  = palette["ped"]
seg[bike_mask] = palette["bike"]

overlay = (0.45 * ortho + 0.55 * seg).astype("uint8")
Image.fromarray(seg).save("paris_voirie_labels.png")
Image.fromarray(overlay).save("paris_voirie_overlay.png")

px = lambda m: int(m.sum() * PX**2)
aoi = (maxx - minx) * (maxy - miny)
print(f"\n=== Paris voirie segmentation ({(maxx-minx):.0f}m x {(maxy-miny):.0f}m AOI) ===")
for name, m in [("Car carriageway", car_mask), ("Pedestrian", ped_mask),
                ("Bike lane", bike_mask)]:
    a = px(m); print(f"  {name:20s} {a:9,d} m²  ({a/aoi*100:5.2f}%)")
print("Saved paris_voirie_labels.png  paris_voirie_overlay.png")
