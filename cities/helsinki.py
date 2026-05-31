"""Pixel-wise segmentation using Helsinki's YLRE engineering-grade polygons.

YLRE = Yleisten alueiden rekisteri (Public Area Register). It is the city's
operational dataset of every public-area surface, built from the surveyed
katusuunnitelmat (street construction plans) and used for billing
maintenance, snow ploughing, repaving, etc. Polygons separate:

  Katu_ja_viherosat_ajorata_alue       -> motor carriageway
  Katu_ja_viherosat_kevytliikenne_alue -> light traffic (sidewalk / bike)
  Katu_ja_viherosat_eiliikenne_alue    -> non-traffic (islands, green strips)

AOI: central Helsinki, spanning Ruoholahti (west) - Pasila (north) -
Kalasatama (east). Coordinates are ETRS-GK25 (EPSG:3879).
"""
import io, math, numpy as np, requests, geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
from PIL import Image

# --- Area Of Interest -------------------------------------------------------
# Tight zoom on Helsinki's southern peninsula, bounded by:
#   west  - Ruoholahti           (~25495000, 6672400)
#   north - Central Railway Stn  (~25496800, 6673200)
#   east  - Kauppatori port      (~25497800, 6672400)
# South edge follows the Hernesaari/Eira shoreline.
# Coordinates are ETRS-GK25 (EPSG:3879), Helsinki's local cadastral CRS.
minx, miny = 25494800, 6671500   # SW corner, metres (E, N)
maxx, maxy = 25498000, 6673400   # NE corner, metres (E, N)

# PX = ground resolution in metres per pixel.
#   YLRE polygons are decimetre-accurate, the orthophoto is 5 cm/px native.
#   PX=0.5 keeps the render below ~50 MP (raster grids are W*H pixels).
#   Memory scales as 1/PX**2, so dropping to 0.05 would mean ~5 GB of RGB.
PX = 0.5

# Cap output dimensions at MAX_PX (per side) so PNGs stay manageable.
MAX_PX = 6000
PX = max(PX, (maxx - minx) / MAX_PX, (maxy - miny) / MAX_PX)

W = int(round((maxx - minx) / PX)); H = int(round((maxy - miny) / PX))

# Affine pixel->world transform (top-left origin, y axis flipped vs. world).
transform = from_origin(minx, maxy, PX, PX)
print(f"AOI {(maxx-minx):.0f} x {(maxy-miny):.0f} m  ->  {W} x {H} px @ {PX} m/px")

# Helsinki Open Data geoservices (kartta.hel.fi). CC BY 4.0.
WMS = "https://kartta.hel.fi/ws/geoserver/avoindata/wms"
WFS = "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"

def wms(layer, tile_px=2048):
    """Download `layer` as a mosaic. We tile because the WMS server rejects
    single GetMap requests above a few thousand pixels per side."""
    out = np.zeros((H, W, 3), "uint8")
    nx = math.ceil(W / tile_px); ny = math.ceil(H / tile_px)
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * tile_px; y0 = iy * tile_px
            tw = min(tile_px, W - x0); th = min(tile_px, H - y0)
            tminx = minx + x0 * PX;   tmaxx = tminx + tw * PX
            tmaxy = maxy - y0 * PX;   tminy = tmaxy - th * PX
            r = requests.get(WMS, params={"service":"WMS","version":"1.1.1","request":"GetMap",
                "layers":layer,"styles":"","srs":"EPSG:3879",
                "bbox":f"{tminx},{tminy},{tmaxx},{tmaxy}",
                "width":tw,"height":th,"format":"image/png"}, timeout=600)
            r.raise_for_status()
            out[y0:y0+th, x0:x0+tw] = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
            print(f"  wms tile {iy*nx+ix+1}/{nx*ny}")
    return out

def wfs(layer):
    """Fetch vector features of `layer` clipped to the AOI bbox."""
    r = requests.get(WFS, params={"service":"WFS","version":"2.0.0","request":"GetFeature",
        "typeNames":layer,"srsName":"EPSG:3879",
        "bbox":f"{minx},{miny},{maxx},{maxy},EPSG:3879",
        "outputFormat":"application/json","count":50000}, timeout=180)
    r.raise_for_status(); return gpd.read_file(io.BytesIO(r.content))

print("Fetching orthophoto...")
ortho = wms("avoindata:Ortoilmakuva_2024_5cm")

# The three top-level YLRE families. Together they tile every surveyed
# public surface in the AOI, with the polygons not overlapping within a
# family (a small amount of overlap can occur across families on shared
# edges — handled with a priority order below).
#   ajorata_alue        -> "carriageway area" (motor vehicles)
#   kevytliikenne_alue  -> "light traffic" (sidewalks + bike lanes)
#   eiliikenne_alue     -> "non-traffic" (medians, green strips, plazas)
LAYERS = {
    "car":    "avoindata:YLRE_Katu_ja_viherosat_ajorata_alue",
    "light":  "avoindata:YLRE_Katu_ja_viherosat_kevytliikenne_alue",
    "nontrf": "avoindata:YLRE_Katu_ja_viherosat_eiliikenne_alue",
}
gdfs = {}
for k, lyr in LAYERS.items():
    g = wfs(lyr)
    print(f"{k:7s}: {len(g):4d} features  cols={list(g.columns)[:6]}")
    gdfs[k] = g

# Print the available subtype values so we can refine the light-traffic split.
if "paatyyppi" in gdfs["light"].columns:
    print("\nlight-traffic paatyyppi:", gdfs["light"]["paatyyppi"].value_counts().to_dict())
if "alatyyppi" in gdfs["light"].columns:
    print("light-traffic alatyyppi (top10):",
          gdfs["light"]["alatyyppi"].value_counts().head(10).to_dict())

def rast(geoms):
    """Burn polygons onto the 3000x3000 pixel grid -> binary mask."""
    geoms=[x for x in geoms if x is not None and not x.is_empty]
    return rasterize([(x,1) for x in geoms], out_shape=(H,W), transform=transform,
                     fill=0, dtype="uint8") if geoms else np.zeros((H,W),"uint8")

# Build masks with priority: car > light > non-traffic. This avoids double
# counting where polygons in different layers overlap on shared edges.
car_mask    = rast(gdfs["car"].geometry).astype(bool)
light_mask  = rast(gdfs["light"].geometry).astype(bool) & ~car_mask
nontrf_mask = rast(gdfs["nontrf"].geometry).astype(bool) & ~car_mask & ~light_mask

# YLRE's light-traffic family lumps cycle and pedestrian polygons together;
# we split them using the Finnish `alatyyppi` (sub-type) string:
#   "pyörä..."  - bike (pyörätie / pyöräilyalue)
#   "jalka.../käytävä/portaat/aukio..." - pedestrian (jalkakäytävä,
#       puistokäytävä, portaat, aukio)
# A polygon that mentions both pyörä AND jalka (e.g. "Yhdistetty jk ja pp"
# = shared ped+bike path) is counted as pedestrian — the conservative choice.
bike_mask = np.zeros((H,W), bool); ped_mask = np.zeros((H,W), bool)
if "alatyyppi" in gdfs["light"].columns:
    light = gdfs["light"]
    bike_kw = ("pyörä",)
    ped_kw  = ("jalka", "käytä", "porras", "aukio")
    def cat(s):
        s = str(s).lower()
        if any(k in s for k in bike_kw) and not any(k in s for k in ped_kw): return "bike"
        return "ped"
    light = light.assign(_cat=light["alatyyppi"].fillna("").map(cat))
    bike_mask = rast(light[light._cat=="bike"].geometry).astype(bool) & ~car_mask
    ped_mask  = rast(light[light._cat=="ped" ].geometry).astype(bool) & ~car_mask & ~bike_mask

# --- Render -----------------------------------------------------------------
palette = {
    "car":    (220,  30,  30),  # red
    "ped":    ( 30,  90, 230),  # blue
    "bike":   (255, 200,   0),  # yellow
    "nontrf": (120, 200,  90),  # green
}
seg = np.full_like(ortho, 255)  # white background — only car/ped/bike are coloured
seg[car_mask]  = palette["car"]
seg[ped_mask]  = palette["ped"]
seg[bike_mask] = palette["bike"]

overlay = (0.45*ortho + 0.55*seg).astype("uint8")
Image.fromarray(seg).save("helsinki_center_ylre_labels.png")
Image.fromarray(overlay).save("helsinki_center_ylre_overlay.png")

# Pixel count -> m² (each pixel is PX*PX m).
px = lambda m: int(m.sum() * PX**2)
aoi_area = (maxx - minx) * (maxy - miny)
print(f"\n=== Engineering-grade segmentation ({(maxx-minx):.0f}m x {(maxy-miny):.0f}m AOI) ===")
for name, m in [("Car carriageway", car_mask), ("Pedestrian", ped_mask),
                ("Bike lane", bike_mask), ("Non-traffic (island/green)", nontrf_mask)]:
    a = px(m); print(f"  {name:30s} {a:9,d} m²  ({a/aoi_area*100:5.2f}%)")
print("Saved helsinki_center_ylre_labels.png  helsinki_center_ylre_overlay.png")
