"""
Crystal geometry loader.

Reads data/crystal_geometry.csv (derived from data/crystal_map_raw.csv)
and provides fast lookup from crystalId to (disk, x, y) in the disk-local frame.
"""

import csv
from pathlib import Path

# Default path relative to project root
_DEFAULT_CSV = Path(__file__).parents[2] / "data" / "crystal_geometry.csv"
_DEFAULT_NEIGHBORS_CSV = Path(__file__).parents[2] / "data" / "crystal_neighbors.csv"


def load_crystal_map(csv_path=None):
    """
    Load crystalId -> (diskId, x_mm, y_mm) from the geometry CSV.

    Returns
    -------
    dict[int, tuple[int, float, float]]
        Mapping from global crystalId to (diskId, x_mm, y_mm).
    """
    path = Path(csv_path or _DEFAULT_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"Crystal geometry file not found: {path}\n"
            "Run the geometry dump first (see src/geometry/README.md)."
        )
    crystal_map = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["crystalId"])
            crystal_map[cid] = (int(row["diskId"]), float(row["x_mm"]), float(row["y_mm"]))
    return crystal_map


def load_neighbor_map(csv_path=None):
    """
    Load crystalId -> list[neighbor_crystalId] from the neighbors CSV.

    Includes both immediate neighbors and next-ring neighbors
    (neighbors() + nextNeighbors() from the Offline geometry service).

    Returns
    -------
    dict[int, list[int]]
    """
    path = Path(csv_path or _DEFAULT_NEIGHBORS_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"Crystal neighbors file not found: {path}\n"
            "Run the geometry dump first (see src/geometry/README.md)."
        )
    neighbor_map = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["crystalId"])
            neighbors = [int(n) for n in row["neighbors"].split(";") if n]
            neighbor_map[cid] = neighbors
    return neighbor_map
