# EU cities — public space, segmented

Pixel-wise renders of how the public realm in European city centres is
allocated between cars, pedestrians, and bikes, built from each city's
own open geodata where possible (surveyed cadastral surface polygons)
and from OpenStreetMap where no such dataset is openly served.

Each script in [`cities/`](cities/) downloads its own data, rasterises
the polygons onto a metric grid, and writes two PNGs:

  * `<city>_<source>_labels.png` — clean colour-coded segmentation
    (red car, blue pedestrian, yellow bike, white background).
  * `<city>_<source>_overlay.png` — the same colours blended 55:45 with
    an aerial photo for context.

## Data sources

| City | Source | Surveyed polygons? |
|---|---|---|
| Helsinki | YLRE via `kartta.hel.fi` | yes (car / ped / bike) |
| Vienna | FMZK via `data.wien.gv.at` | yes (car / ped); bike from buffered lines |
| Milano | DBT 2012 via Comune di Milano ArcGIS | yes (car / ped / bike) + OSM ped override |
| Paris | Plan de voirie via `opendata.paris.fr` | yes (sidewalk / bike); car = envelope minus the rest |
| Bern, Zurich | Swiss AV (`geodienste.ch/db/av_0/deu`) | yes (car / ped); Zurich adds bike via Stadt Zürich Velonetz |
| All other cities | OpenStreetMap via Overpass | no — buffered centerlines |

OSM-fallback numbers are estimates: roads are widened by class-specific
typical widths, not measured surfaces.

## Animations

Two cities with open *traffic count* data have an animation in
[`animations/`](animations/):

  * `milano_animation.py` — 42 Area C gates, 30-min counts, one day.
  * `bologna_animation.py` — 356 spire (induction loops) inside the
    centre, hourly counts, one day.

Both scripts download everything on first run and cache intermediates
under `~/.cache/`.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cities/helsinki.py        # produces helsinki_center_ylre_*.png
python animations/milano_animation.py
```

Outputs land in the current working directory by default. Heavy
binaries (overlays, animations) are gitignored — commit only what you
intend to publish.

## Poster

[`poster/cities_poster.tex`](poster/cities_poster.tex) lays the 16 most
illustrative city renders into a single A0 portrait poster. Build with
`pdflatex cities_poster.tex` after generating the `*_labels.png`
files referenced in the .tex.
