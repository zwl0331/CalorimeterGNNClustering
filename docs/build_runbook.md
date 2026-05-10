# Build & Reprocessing Runbook

Operational notes for the C++/Offline build (muse) and ROOT-file reprocessing pipeline. Pulled out of `CLAUDE.md` to keep that file focused on Python development. Read this when rebuilding EventNtuple, regenerating v2 ROOT files, or submitting FermiGrid reprocessing jobs.

## Mu2e build environment (muse)

For building C++ packages (EventNtuple, Offline, Production), `mu2einit` and `muse` are interactive shell functions. On AlmaLinux 9, `setupmu2e-art.sh` handles spack + muse setup automatically — no separate `$SPACK_ROOT` sourcing needed:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
cd /exp/mu2e/app/users/wzhou2/working_dir
muse setup
muse build -j64   # use many cores on build nodes
```

Build location: `/exp/mu2e/app/users/wzhou2/working_dir/` — local checkouts of `EventNtuple/`, `Offline/`, `Production/`. Modified EventNtuple adds the `calomcsim.ancestorSimIds` branch for SimParticle ancestry.

## Muse build gotchas (discovered 2026-04-04)

- `--TFileName` does NOT work with the `mu2e` command — use `-T` (short flag) instead.
- EventNtuple `getTrigPathNameByIndex` → `getTrigPathName`: API renamed in current CVMFS Offline. Already fixed in local checkout.
- `from_mcs-mockdata.fcl` requires the `ArtAnalysis` package (for TrkQual/TrkPID). Use `from_mcs-calo-only.fcl` to skip track analysis and avoid this dependency.
- `Production` repo must be cloned into the working dir for `epilog.fcl` dependency.
- `epilog.fcl` sets `TFileService.fileName: "/dev/null"` by default — FCL lines after the include override this.

## Muse tarball

Latest tarball at `/exp/mu2e/data/users/wzhou2/museTarball/tmp.iTQSPdWZBd/Code.tar.bz2` (rebuilt 2026-04-13, `al9-prof-e29-p092`, with `ancestorSimIds` + latest Offline). Used for FermiGrid jobs.

## Batch reprocessing (local)

Running the local-batch reprocessing pipeline (used when v2 ROOT files need regeneration on cluster nodes rather than via FermiGrid):

- `run_all.sh` uses `MAX_PARALLEL=10`. Originally 20, but the login node killed all processes after ~10 min (resource limits). 10 parallel is safe.
- Each file takes ~8–10 min. Full 50-file batch: ~40–50 min with 10 parallel.
- Script skips files that already exist — safe to rerun after partial failures. Delete partial outputs first (check file sizes; complete files are ~1.9–2.0 GB).
- Must run inside tmux with the muse environment:
  ```bash
  tmux new -s reprocess
  source setupmu2e-art.sh && cd working_dir && muse setup && bash run_all.sh
  ```

## FermiGrid submission

For larger reprocessing jobs:

- **MDC2025:** `root_files_v2/grid_submit.sh`. Original cluster `90854576`. Train split was deleted 2026-04-13 to free quota; reproducible via this grid path if needed.
- **Run1B:** `root_files_run1b/grid_submit.sh`, workflow project `eventntuple_run1b`. Cluster `27583402`.
