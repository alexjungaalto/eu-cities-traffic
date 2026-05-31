"""Pixel-wise segmentation of Vienna's city centre using FMZK polygons.

FMZK = Flächen-Mehrzweckkarte (Vienna's surveyed, decimetre-accurate base
map of all surface polygons). Open data on data.wien.gv.at. Layers used:

  ogdwien:FMZKVERKEHR1OGD   -> 'Fahrbahn' (motor carriageway)
  ogdwien:FMZKVERKEHR2OGD   -> 'Gehsteig' (sidewalk)
  ogdwien:RADWEGEOGD        -> cycling infrastructure (LineString, buffered)

Coordinates are MGI/Austria GK East (EPSG:31256). Orthophoto from
ogdwien WMS layer GRM2023OGD.
"""
import io, math, numpy as np, requests, geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
from PIL import Image

# --- Area Of Interest -------------------------------------------------------
# Innere Stadt extended westward to include Westbahnhof (Europaplatz).
# Coordinates are MGI / Austria GK East (EPSG:31256).
# Reference points: Westbahnhof ~ (420, 339780); Stephansdom ~ (3080, 341000).
minx, miny =  -200, 339700   # SW corner in metres (E, N); west of Westbahnhof
maxx, maxy =  4400, 342500   # NE corner — Donaukanal / Ringstrasse east edge

# PX = ground resolution in metres per pixel of the rasterised output.
#   PX=0.5  -> a 1 m x 1 m patch becomes a 2x2 block of pixels.
#   Smaller PX = sharper map but quadratically more memory / WMS load.
#   At 0.5 the 2.6 x 2.8 km AOI becomes 5200 x 5600 px ~ 87 MB RGB in RAM.
PX = 0.5

# Cap output dimensions at MAX_PX so PNGs stay viewable. If the AOI at the
# requested PX would exceed it, coarsen PX automatically.
MAX_PX = 6000
PX = max(PX, (maxx - minx) / MAX_PX, (maxy - miny) / MAX_PX)

# Raster dimensions and the affine transform that maps pixel (col,row) ->
# world (x,y). from_origin(top-left x, top-left y, pixel_size_x, pixel_size_y)
# is the rasterio convention: y decreases downward in image space.
W = int(round((maxx - minx) / PX)); H = int(round((maxy - miny) / PX))
transform = from_origin(minx, maxy, PX, PX)
print(f"AOI {(maxx-minx):.0f} x {(maxy-miny):.0f} m  ->  {W} x {H} px @ {PX} m/px")

# --- Open Government Data Wien endpoints ------------------------------------
WMS = "https://data.wien.gv.at/daten/wms"   # rendered raster tiles (PNG)
WFS = "https://data.wien.gv.at/daten/geo"   # vector features (GeoJSON)

def wms(layer, tile_px=2048):
    """Download a rendered raster of `layer` covering the full AOI.

    The WMS server caps response size, so we tile the request: each call
    fetches at most `tile_px` x `tile_px` pixels, then we mosaic them
    into a single (H, W, 3) RGB array.
    """
    out = np.zeros((H, W, 3), "uint8")
    nx = math.ceil(W / tile_px); ny = math.ceil(H / tile_px)
    for iy in range(ny):
        for ix in range(nx):
            # Pixel offset of this tile inside the output mosaic.
            x0 = ix * tile_px; y0 = iy * tile_px
            tw = min(tile_px, W - x0); th = min(tile_px, H - y0)
            # World bbox for the tile (note y axis flip: row 0 is north).
            tminx = minx + x0 * PX;   tmaxx = tminx + tw * PX
            tmaxy = maxy - y0 * PX;   tminy = tmaxy - th * PX
            r = requests.get(WMS, params={"service":"WMS","version":"1.1.1","request":"GetMap",
                "layers":layer,"styles":"","srs":"EPSG:31256",
                "bbox":f"{tminx},{tminy},{tmaxx},{tmaxy}",
                "width":tw,"height":th,"format":"image/png"}, timeout=600)
            r.raise_for_status()
            out[y0:y0+th, x0:x0+tw] = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
            print(f"  wms tile {iy*nx+ix+1}/{nx*ny}")
    return out

def wfs(layer):
    """Download vector features of `layer` clipped to the AOI bbox.

    `count=200000` is just a safety upper bound — well above the number
    of FMZK polygons the Innere Stadt actually contains.
    """
    r = requests.get(WFS, params={"service":"WFS","version":"2.0.0","request":"GetFeature",
        "typeNames":layer,"srsName":"EPSG:31256",
        "bbox":f"{minx},{miny},{maxx},{maxy},EPSG:31256",
        "outputFormat":"application/json","count":200000}, timeout=600)
    r.raise_for_status(); return gpd.read_file(io.BytesIO(r.content))

print("Fetching orthophoto (GRM2023OGD)...")
ortho = wms("GRM2023OGD")

print("Fetching FMZK carriageway (FMZKVERKEHR1OGD)...")
verk1 = wfs("ogdwien:FMZKVERKEHR1OGD")
print(f"  {len(verk1)} features  layers={verk1['LAYER'].value_counts().to_dict()}")
print("Fetching FMZK sidewalk (FMZKVERKEHR2OGD)...")
verk2 = wfs("ogdwien:FMZKVERKEHR2OGD")
print(f"  {len(verk2)} features  layers={verk2['LAYER'].value_counts().to_dict()}")
print("Fetching cycling lines (RADWEGEOGD)...")
rad = wfs("ogdwien:RADWEGEOGD")
print(f"  {len(rad)} features  MERKMAL={rad['MERKMAL'].value_counts().to_dict()}")

# --- Pick the polygon sub-types we want per mobility mode -------------------
# FMZKVERKEHR1 holds many sub-classes via the LAYER attribute. We treat as
# "car surface" anything the motor vehicle network actually uses:
#   Fahrbahn = carriageway; Ruhender Verkehr = parking; Zebrastreifen =
#   pedestrian crossings (cars still drive over them); Private
#   Verkehrsfläche = private driveways; Fahrbahnschwelle = speed table.
car_layers = {"Fahrbahn", "Ruhender Verkehr", "Zebrastreifen",
              "Private Verkehrsfläche", "Fahrbahnschwelle"}
fahrbahn = verk1[verk1["LAYER"].isin(car_layers)]

# FMZKVERKEHR2 is "everything else paved that isn't carriageway." The union
# below approximates what a pedestrian can legally walk on at street level.
ped_layers = {"Gehsteig", "Befestigte Fläche", "Gehweg, Radweg",
              "Stiege", "Innenhof", "Sonstige Verkehrsfläche", "Portal",
              "Fußgängerzone"}
gehsteig = verk2[verk2["LAYER"].isin(ped_layers)]

# Bike infra is published as LineString geometry only — we buffer each line
# by 1.25 m on each side (~2.5 m total width), the typical legal width of
# a Radfahrstreifen / baulicher Radweg in Vienna. We exclude "Radroute"
# (signed wayfinding only — usually a shared lane, no dedicated surface).
bike_kinds = {"Getrennte Führung", "Markierte Anlagen", "Mehrzweckstreifen",
              "Radfahrstreifen", "Gemischte Führung"}
rad_use = rad[rad["MERKMAL"].isin(bike_kinds)]
bike_polys = rad_use.geometry.buffer(1.25)

def rast(geoms):
    """Burn an iterable of shapely polygons onto the (H, W) raster grid.

    Returns a uint8 array where 1 = polygon covers that pixel, 0 = not.
    The `transform` global links pixel coords to EPSG:31256 metres.
    """
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return rasterize([(g,1) for g in geoms], out_shape=(H,W), transform=transform,
                     fill=0, dtype="uint8") if geoms else np.zeros((H,W),"uint8")

# Priority ordering: car > bike > ped. A pixel that is both carriageway and
# (per the buffered bike line) bike lane is counted as car. Bike claims
# before ped because "Gehweg, Radweg" appears in the ped layer set as well
# — without this ordering the buffered cycle ribbons would be swallowed.
car_mask  = rast(fahrbahn.geometry).astype(bool)
bike_mask = rast(bike_polys).astype(bool) & ~car_mask
ped_mask  = rast(gehsteig.geometry).astype(bool) & ~car_mask & ~bike_mask

# RGB triples for each class; everything else gets painted white.
palette = {"car": (220,30,30), "ped": (30,90,230), "bike": (255,200,0)}
seg = np.full_like(ortho, 255)
seg[car_mask]  = palette["car"]
seg[ped_mask]  = palette["ped"]
seg[bike_mask] = palette["bike"]

# 45/55 alpha blend with the aerial photo for the human-readable overlay.
overlay = (0.45*ortho + 0.55*seg).astype("uint8")
Image.fromarray(seg).save("vienna_fmzk_labels.png")
Image.fromarray(overlay).save("vienna_fmzk_overlay.png")

# Each pixel covers PX*PX m² on the ground; sum the boolean mask to get area.
px = lambda m: int(m.sum() * PX**2)
aoi = (maxx-minx)*(maxy-miny)
print(f"\n=== Vienna FMZK segmentation ({(maxx-minx):.0f}m x {(maxy-miny):.0f}m AOI) ===")
for name, m in [("Car carriageway", car_mask), ("Sidewalk", ped_mask),
                ("Bike (buffered)", bike_mask)]:
    a = px(m); print(f"  {name:20s} {a:9,d} m²  ({a/aoi*100:5.2f}%)")
print("Saved vienna_fmzk_labels.png  vienna_fmzk_overlay.png")
