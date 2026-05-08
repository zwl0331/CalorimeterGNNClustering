"""Task 18b — Stage B inference diagnostics on FlateMinusMixLow-KL.

Run existing SEN + CCN models on MixLow standard NTS files (no ancestry, no
truth-aware metrics). Compare GNN cluster assignments to BFS as a reference,
flag distribution drift in edge features and edge-score saturation.

Diagnostics:
  - Edge logit (raw model output) distribution: detect threshold saturation
    or score collapse vs MDC2025 train regime.
  - Fraction of edges classified positive at tuned threshold.
  - Cluster count per disk: BFS vs SEN vs CCN.
  - Cluster size distribution comparison.
  - Edge-feature distribution after MDC2025 normalization: detect z-score
    drift (model is being asked to interpret OOD inputs).
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import uproot
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_builder import build_graph, compute_edge_features, compute_node_features
from src.data.normalization import load_stats, normalize_graph
from src.geometry.crystal_geometry import load_crystal_map
from src.inference.cluster_reco import reconstruct_clusters
from src.models import build_model
from torch_geometric.data import Data


NTS_BRANCHES = [
    "calohits.crystalId_",
    "calohits.eDep_",
    "calohits.time_",
    "calohits.clusterIdx_",
]


def percentiles(arr, qs=(0.05, 0.5, 0.95, 0.99)):
    if len(arr) == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.quantile(arr, q)) for q in qs}


def fmt_dist(arr, name, prec=2):
    if len(arr) == 0:
        return f"  {name}: (empty)"
    p = percentiles(arr)
    return (
        f"  {name}:  n={len(arr):>7}  "
        f"mean={np.mean(arr):>{prec+5}.{prec}f}  "
        f"p05={p[0.05]:>{prec+5}.{prec}f}  "
        f"med={p[0.5]:>{prec+5}.{prec}f}  "
        f"p95={p[0.95]:>{prec+5}.{prec}f}  "
        f"p99={p[0.99]:>{prec+5}.{prec}f}"
    )


def load_model(config_path, checkpoint_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    inf_cfg = cfg["inference"]
    tau_edge = inf_cfg["tau_edge"]
    has_node_head = cfg["model"].get("name", "SimpleEdgeNet") == "CaloClusterNet"
    lambda_node = cfg.get("train", {}).get("lambda_node", 0.0)
    tau_node = inf_cfg.get("tau_node") if (has_node_head and lambda_node > 0) else None
    return model, cfg, tau_edge, tau_node


def n_clusters_from_labels(labels):
    """Count distinct nonneg cluster IDs among hit labels."""
    arr = np.asarray(labels)
    if arr.size == 0:
        return 0
    valid = arr[arr >= 0]
    return int(len(np.unique(valid)))


def cluster_sizes(labels):
    arr = np.asarray(labels)
    valid = arr[arr >= 0]
    if valid.size == 0:
        return []
    _, counts = np.unique(valid, return_counts=True)
    return counts.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixlow-dir", type=Path,
        default=Path("/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMixLow-KL/Run1B-004/root"),
    )
    parser.add_argument("--n-files", type=int, default=3)
    parser.add_argument("--max-events-per-file", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/run1b_mixlow_stageB")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load both v2 models
    models = {}
    for name, cfg_path, ckpt_path in [
        ("SEN",
         "configs/default.yaml",
         "outputs/runs/simple_edge_net_v2/checkpoints/best_model.pt"),
        ("CCN",
         "configs/calo_cluster_net_saliency.yaml",
         "outputs/runs/calo_cluster_net_v2_saliency/checkpoints/best_model.pt"),
    ]:
        model, cfg, tau_edge, tau_node = load_model(cfg_path, ckpt_path, device)
        models[name] = {"model": model, "cfg": cfg,
                        "tau_edge": tau_edge, "tau_node": tau_node}
        print(f"  {name}: tau_edge={tau_edge}, tau_node={tau_node}")

    graph_cfg = models["SEN"]["cfg"]["graph"]
    stats = load_stats(models["SEN"]["cfg"]["data"]["normalization_stats"])
    crystal_map = load_crystal_map("data/crystal_geometry.csv")

    files = sorted(args.mixlow_dir.rglob("*.root"))[: args.n_files]
    print(f"\nMixLow files: {len(files)}")
    print(f"Max events per file: {args.max_events_per_file}\n")

    # Diagnostic accumulators
    edge_logits_all = {"SEN": [], "CCN": []}     # raw scores per edge
    edge_pos_frac = {"SEN": [], "CCN": []}        # frac positive per disk-graph
    n_clusters_per_disk = {"BFS": [], "SEN": [], "CCN": []}
    cluster_sizes_all = {"BFS": [], "SEN": [], "CCN": []}
    edge_feat_norm_per_dim = [[] for _ in range(8)]   # post-norm distributions

    n_disk_graphs = 0
    n_events_total = 0
    t0 = time.time()

    for fi, fpath in enumerate(files):
        print(f"  [{fi+1}/{len(files)}] {fpath.name}...", flush=True)
        tree = uproot.open(str(fpath) + ":EventNtuple/ntuple")
        arrays = tree.arrays(NTS_BRANCHES, entry_stop=args.max_events_per_file)
        n_events = len(arrays)
        n_events_total += n_events

        for ev in range(n_events):
            cryids = np.array(arrays["calohits.crystalId_"][ev], dtype=np.int64)
            n_total = len(cryids)
            if n_total == 0:
                continue
            energies = np.array(arrays["calohits.eDep_"][ev], dtype=np.float64)
            times = np.array(arrays["calohits.time_"][ev], dtype=np.float64)
            cidx = np.array(arrays["calohits.clusterIdx_"][ev], dtype=np.int64)

            # Resolve disk + (x, y) from crystal_map (standard NTS lacks crystalPos_)
            xs = np.zeros(n_total, dtype=np.float64)
            ys = np.zeros(n_total, dtype=np.float64)
            disks = np.full(n_total, -1, dtype=np.int64)
            for i, c in enumerate(cryids):
                c = int(c)
                if c in crystal_map:
                    disks[i], xs[i], ys[i] = crystal_map[c]

            for disk_id in (0, 1):
                m = disks == disk_id
                n_disk = int(m.sum())
                if n_disk < 2:
                    continue

                d_pos = np.stack([xs[m], ys[m]], axis=1)
                d_t = times[m]
                d_e = energies[m]
                d_cidx = cidx[m]

                edge_index, _ = build_graph(
                    d_pos, d_t,
                    r_max=graph_cfg["r_max_mm"],
                    dt_max=graph_cfg["dt_max_ns"],
                    k_min=graph_cfg["k_min"],
                    k_max=graph_cfg["k_max"],
                )
                if edge_index.shape[1] == 0:
                    continue

                node_feat = compute_node_features(d_pos, d_t, d_e)
                edge_feat = compute_edge_features(d_pos, d_t, d_e, edge_index)
                data = Data(
                    x=torch.from_numpy(node_feat),
                    edge_index=torch.from_numpy(edge_index),
                    edge_attr=torch.from_numpy(edge_feat),
                )
                normalize_graph(data, stats)

                # Sample post-norm edge features (subsample to keep memory bounded)
                ea = data.edge_attr.numpy()
                for d in range(8):
                    edge_feat_norm_per_dim[d].extend(ea[:, d].tolist())

                # BFS clusters from clusterIdx_
                n_clusters_per_disk["BFS"].append(n_clusters_from_labels(d_cidx))
                cluster_sizes_all["BFS"].extend(cluster_sizes(d_cidx))

                # Run both GNNs
                for name in ("SEN", "CCN"):
                    m_info = models[name]
                    with torch.no_grad():
                        out = m_info["model"](data.to(device))
                    if isinstance(out, dict):
                        logits = out["edge_logits"].cpu().numpy()
                        nl = out.get("node_logits")
                        node_logits = nl.cpu().numpy() if nl is not None else None
                    else:
                        logits = out.cpu().numpy()
                        node_logits = None

                    scores = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
                    edge_logits_all[name].extend(scores.tolist())

                    # Both directions of each undirected edge are present in
                    # edge_index; count positive fraction over directed edges
                    # (consistent with how the model and the cluster reco use it).
                    pos_frac = float((scores > m_info["tau_edge"]).mean())
                    edge_pos_frac[name].append(pos_frac)

                    labels, _ = reconstruct_clusters(
                        edge_index=edge_index,
                        edge_logits=logits,
                        n_nodes=n_disk,
                        energies=d_e,
                        tau_edge=m_info["tau_edge"],
                        min_hits=1,
                        min_energy_mev=0.0,
                        node_logits=node_logits,
                        tau_node=m_info["tau_node"],
                    )
                    n_clusters_per_disk[name].append(n_clusters_from_labels(labels))
                    cluster_sizes_all[name].extend(cluster_sizes(labels))

                n_disk_graphs += 1

    elapsed = time.time() - t0
    print(f"\nProcessed {n_events_total} events → {n_disk_graphs} disk-graphs in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Diagnostics output
    # ------------------------------------------------------------------
    print("\n=== Edge sigmoid scores (raw model output, all directed edges) ===")
    for name in ("SEN", "CCN"):
        arr = np.asarray(edge_logits_all[name])
        n_pos = int((arr > models[name]["tau_edge"]).sum())
        print(f"  {name} (tau={models[name]['tau_edge']}):")
        print(fmt_dist(arr, "score", prec=4))
        print(f"    edges>tau: {n_pos:>9} / {len(arr):>9} ({100*n_pos/max(len(arr),1):.2f}%)")

    print("\n=== Per-disk-graph positive-edge fraction ===")
    for name in ("SEN", "CCN"):
        arr = np.asarray(edge_pos_frac[name])
        print(fmt_dist(arr, f"{name} pos-frac", prec=3))

    print("\n=== Clusters per disk (cluster reco) ===")
    for name in ("BFS", "SEN", "CCN"):
        print(fmt_dist(np.asarray(n_clusters_per_disk[name]), f"{name:<3}", prec=1))

    print("\n=== Cluster sizes (hits per cluster) ===")
    for name in ("BFS", "SEN", "CCN"):
        print(fmt_dist(np.asarray(cluster_sizes_all[name]), f"{name:<3}", prec=1))

    print("\n=== Post-normalization edge features (z-score should be ~ N(0,1)) ===")
    feat_names = ["dx", "dy", "dist", "dt", "dlogE", "Easym", "logSumE", "dr"]
    for d, nm in enumerate(feat_names):
        arr = np.asarray(edge_feat_norm_per_dim[d])
        print(fmt_dist(arr, f"feat[{d}] {nm:<7}", prec=2))

    # ------------------------------------------------------------------
    # Save raw arrays for plotting later
    # ------------------------------------------------------------------
    np.savez(
        args.output_dir / "stageB_diagnostics.npz",
        edge_scores_SEN=np.asarray(edge_logits_all["SEN"]),
        edge_scores_CCN=np.asarray(edge_logits_all["CCN"]),
        edge_pos_frac_SEN=np.asarray(edge_pos_frac["SEN"]),
        edge_pos_frac_CCN=np.asarray(edge_pos_frac["CCN"]),
        n_clusters_BFS=np.asarray(n_clusters_per_disk["BFS"]),
        n_clusters_SEN=np.asarray(n_clusters_per_disk["SEN"]),
        n_clusters_CCN=np.asarray(n_clusters_per_disk["CCN"]),
        sizes_BFS=np.asarray(cluster_sizes_all["BFS"]),
        sizes_SEN=np.asarray(cluster_sizes_all["SEN"]),
        sizes_CCN=np.asarray(cluster_sizes_all["CCN"]),
        **{f"edge_feat_norm_{d}": np.asarray(edge_feat_norm_per_dim[d]) for d in range(8)},
    )
    print(f"\nSaved raw arrays to {args.output_dir / 'stageB_diagnostics.npz'}")


if __name__ == "__main__":
    main()
