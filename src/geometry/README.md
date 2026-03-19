# Crystal Geometry Dump (Task 0)

This directory will contain a one-time C++ art module that reads the Mu2e
calorimeter geometry service and writes `data/crystal_geometry.csv` and
`data/crystal_neighbors.csv`.

## Why this is needed

The EventNtuple `calohits` branch stores only `crystalId_` — no (x,y) positions.
Crystal positions come from the Mu2e geometry service (C++ only, via `GeomHandle`).

## What will be produced

`data/crystal_geometry.csv`:
```
crystalId,diskId,x_mm,y_mm
0,0,-417.7,273.7
1,0,...
...
```

`data/crystal_neighbors.csv`:
```
crystalId,neighbors
0,1;2;3;4
1,0;2;5;...
...
```
Neighbors = immediate ring + next ring (`neighbors()` + `nextNeighbors()` from Offline).

## How to run (once, requires muse setup Offline)

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
cd /exp/mu2e/app/users/wzhou2/working_dir
muse setup Offline
mu2e -c /path/to/this/project/src/geometry/dump_crystal_geometry.fcl
```

The output CSV files go to `data/` and should be committed to the repository.
They are static for a given detector configuration (MDC2020 geometry).

## Files to create (pending)

- `DumpCrystalGeometry_module.cc` — art EDAnalyzer that loops `cal.nCrystals()`
- `dump_crystal_geometry.fcl` — FCL to run it with standard services
