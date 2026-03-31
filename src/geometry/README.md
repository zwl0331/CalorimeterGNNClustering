# Crystal Geometry (Task 0)

This module provides crystal position and neighbor lookups for the Mu2e calorimeter.

## Data files

- `data/crystal_map_raw.csv` — Full crystal/SiPM channel map (2740 rows). Source of truth.
- `data/crystal_geometry.csv` — Derived: `crystalId,diskId,x_mm,y_mm` for 1348 crystals (1344 CAL + 4 CAPHRI).
- `data/crystal_neighbors.csv` — Derived: `crystalId,neighbors` (immediate geometric neighbors within 1.5× crystal pitch ≈ 51.5 mm).

All three files are static for the MDC2020 geometry and committed to the repo.

## CAPHRI crystals

4 crystal bars (IDs 582, 609, 610, 637) are **not CsI** — they are CAPHRI type, all on disk 0.

## Python API

```python
from src.geometry.crystal_geometry import load_crystal_map, load_neighbor_map

crystal_map = load_crystal_map()    # {crystalId: (diskId, x_mm, y_mm)}
neighbor_map = load_neighbor_map()  # {crystalId: [neighbor_ids]}
```
