"""Pixel-wise segmentation of central Zurich.

Carriageway and pedestrian polygons come from the Swiss Amtliche
Vermessung (AV) Bodenbedeckung dataset.

AV is Switzerland's cadastral surface survey — every parcel and surface
classified at decimetre accuracy. Bodenbedeckung ("land cover") polygons
carry an `Art` attribute that names the surface type:

  Strasse_Weg          -> motor carriageway / shared street
  Trottoir             -> sidewalk
  Verkehrsinsel        -> traffic island
  uebrige_befestigte   -> other paved (squares, plazas, courtyards)
  Gebaeude             -> building footprints

Bike lanes are NOT a separate Art in AV. For Zurich we layer on a second
WFS (Stadt Zürich's "Fuss- und Velowegnetz") which has a 'velostreifen'
attribute (painted lane: 'TF'/'FT'/'BOTH') and a 'veloweg' attribute
(segregated path: 1). We buffer those line segments by 1.25 m to get
a ribbon polygon.

WFS sources:
  AV polygons:   geodienste.ch/db/av_0/deu          (layer ms:LCSF)
  bike network:  www.ogd.stadt-zuerich.ch/wfs/...   (layer tbl_routennetz)
CRS: EPSG:2056 LV95. Orthophoto: swisstopo SWISSIMAGE.
"""
import io, math, numpy as np, requests, geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
from PIL import Image

# --- Area Of Interest -------------------------------------------------------
# 4 x 4 km square centred on Zürich Hauptbahnhof / Bahnhofstrasse area
# (~2683100, 1247800 in LV95). Covers the Altstadt (Niederdorf,
# Lindenhof), the Limmat banks, HB rail yards, Kreis 4/5 and a slice of
# the lakefront.
minx, miny = 2681100, 1245800   # SW corner (E, N) in metres
maxx, maxy = 2685100, 1249800   # NE corner

# PX = ground resolution in metres/pixel; see helsinki.py for context.
PX = 0.5
MAX_PX = 6000
PX = max(PX, (maxx - minx) / MAX_PX, (maxy - miny) / MAX_PX)
W = int(round((maxx - minx) / PX)); H = int(round((maxy - miny) / PX))
transform = from_origin(minx, maxy, PX, PX)
print(f"AOI {(maxx-minx):.0f} x {(maxy-miny):.0f} m  ->  {W} x {H} px @ {PX:.3f} m/px")

WFS_BASE = "https://geodienste.ch/db/av_0/deu"
WMS_ORTHO = "https://wms.geo.admin.ch/"
ORTHO_LAYER = "ch.swisstopo.swissimage"


def ortho_image(tile_px=2048):
    """Mosaic SWISSIMAGE WMS tiles into one (H, W, 3) RGB array."""
    out = np.zeros((H, W, 3), "uint8")
    nx = math.ceil(W / tile_px); ny = math.ceil(H / tile_px)
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * tile_px; y0 = iy * tile_px
            tw = min(tile_px, W - x0); th = min(tile_px, H - y0)
            tminx = minx + x0 * PX;   tmaxx = tminx + tw * PX
            tmaxy = maxy - y0 * PX;   tminy = tmaxy - th * PX
            r = requests.get(WMS_ORTHO, params={
                "service":"WMS","version":"1.3.0","request":"GetMap",
                "layers":ORTHO_LAYER,"styles":"",
                "crs":"EPSG:2056",
                # swisstopo's WMS 1.3.0 takes E,N order for EPSG:2056
                # (empirically; N,E returns black PNGs).
                "bbox":f"{tminx},{tminy},{tmaxx},{tmaxy}",
                "width":tw,"height":th,"format":"image/png"}, timeout=600)
            r.raise_for_status()
            out[y0:y0+th, x0:x0+tw] = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
            print(f"  ortho tile {iy*nx+ix+1}/{nx*ny}")
    return out


def _wfs_tile(bb):
    """One WFS GetFeature call (GML output — JSON not supported on this
    server). Returns a GeoDataFrame."""
    r = requests.get(WFS_BASE, params={
        "SERVICE":"WFS","VERSION":"2.0.0","REQUEST":"GetFeature",
        "TYPENAMES":"ms:LCSF","SRSNAME":"EPSG:2056",
        "BBOX":f"{bb[0]},{bb[1]},{bb[2]},{bb[3]},EPSG:2056",
        "COUNT":"5000"}, timeout=600)
    r.raise_for_status()
    return gpd.read_file(io.BytesIO(r.content))


def fetch_av():
    """Fetch AV Bodenbedeckung polygons. We split the AOI into a fixed
    grid because the WFS caps each response at 5000 features."""
    parts = []
    nx = ny = 3   # 9 sub-tiles -> ~500x500 m per tile for the 3 km AOI
    dx = (maxx - minx) / nx; dy = (maxy - miny) / ny
    for i in range(nx):
        for j in range(ny):
            bb = (minx + i*dx, miny + j*dy, minx + (i+1)*dx, miny + (j+1)*dy)
            g = _wfs_tile(bb)
            print(f"  tile {i*ny+j+1}/{nx*ny}: {len(g)} polygons")
            parts.append(g)
    out = gpd.GeoDataFrame(__import__("pandas").concat(parts, ignore_index=True), crs="EPSG:2056")
    # Tile boundaries can yield duplicates for polygons that straddle.
    out = out.drop_duplicates(subset="gml_id")
    return out


print("Fetching SWISSIMAGE orthophoto...")
ortho = ortho_image()

print("Fetching AV Bodenbedeckung (ms:LCSF)...")
av = fetch_av()
print(f"  {len(av)} polygons; Art counts: {av['Art'].value_counts().to_dict()}")

car_polys  = av[av["Art"] == "Strasse_Weg"]
# Sidewalk-equivalent: Trottoir + Verkehrsinsel. We do NOT include
# `uebrige_befestigte` because it lumps in courtyards, garden paths and
# plazas, which would inflate the pedestrian share.
ped_polys  = av[av["Art"].isin(["Trottoir", "Verkehrsinsel"])]

# --- Bike network (Stadt Zürich Fuss- und Velowegnetz) ---------------------
# Segments with velostreifen != '0' (painted lane in any direction) OR
# veloweg == 1 (segregated cycle path) are real bike infrastructure.
# Plain `velo=1` (= bikes allowed) is too inclusive — it just means the
# road is legally cyclable, which is almost every road.
print("Fetching Stadt Zürich bike network (tbl_routennetz)...")
VELO_WFS = "https://www.ogd.stadt-zuerich.ch/wfs/geoportal/Fuss__und_Velowegnetz"
r = requests.get(VELO_WFS, params={
    "SERVICE":"WFS","VERSION":"1.1.0","REQUEST":"GetFeature",
    "TYPENAME":"tbl_routennetz","SRSNAME":"EPSG:2056",
    "BBOX":f"{minx},{miny},{maxx},{maxy},EPSG:2056"}, timeout=600)
r.raise_for_status()
velo = gpd.read_file(io.BytesIO(r.content))
bike_marked = velo["velostreifen"].astype(str).isin({"TF","FT","BOTH","1"})
bike_segregated = velo["veloweg"] == 1
bike_lines = velo[bike_marked | bike_segregated]
print(f"  velo segments: {len(velo)}; bike-relevant: {len(bike_lines)}")
# Buffer each line by 1.25 m -> ~2.5 m wide ribbon, matching the Vienna
# buffer used for Radwege.
bike_polys = bike_lines.geometry.buffer(1.25)


def rast(geoms):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=transform,
                     fill=0, dtype="uint8") if geoms else np.zeros((H, W), "uint8")

# Priority car > bike > ped. A pixel that is both AV-Strasse_Weg and bike
# infrastructure is classified as car (the lane sits inside the carriageway).
# Pedestrian then claims what's left.
car_mask  = rast(car_polys.geometry).astype(bool)
bike_mask = rast(bike_polys).astype(bool) & ~car_mask
ped_mask  = rast(ped_polys.geometry).astype(bool) & ~car_mask & ~bike_mask

palette = {"car": (220, 30, 30), "ped": (30, 90, 230), "bike": (255, 200, 0)}
seg = np.full_like(ortho, 255)
seg[car_mask]  = palette["car"]
seg[ped_mask]  = palette["ped"]
seg[bike_mask] = palette["bike"]

overlay = (0.45 * ortho + 0.55 * seg).astype("uint8")
Image.fromarray(seg).save("zurich_av_labels.png")
Image.fromarray(overlay).save("zurich_av_overlay.png")

px = lambda m: int(m.sum() * PX**2)
aoi = (maxx - minx) * (maxy - miny)
print(f"\n=== Zurich segmentation ({(maxx-minx):.0f}m x {(maxy-miny):.0f}m AOI) ===")
for name, m in [("Car carriageway", car_mask), ("Pedestrian", ped_mask),
                ("Bike (buffered)", bike_mask)]:
    a = px(m); print(f"  {name:20s} {a:9,d} m²  ({a/aoi*100:5.2f}%)")
print("Saved zurich_av_labels.png  zurich_av_overlay.png")
