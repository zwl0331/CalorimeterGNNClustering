"""Primary-level (calo-entrant) truth labeling for GNN calorimeter clustering.

Redefines truth clusters at the calo-entrant level: all hits from the
same electromagnetic shower are assigned to the same truth cluster,
instead of being split by individual SimParticle ID.

The calo-entrant for a SimParticle on disk D is the highest ancestor in
its Geant4 parent chain that also deposited energy in disk D.  If no
ancestor deposited in disk D, the particle itself is the calo-entrant.

This collapses secondary shower products (bremsstrahlung photons, etc.)
into their parent shower, eliminating artificial singleton truth clusters
and reducing ambiguous hit counts.

Requires v2 ROOT files with ``calomcsim.ancestorSimIds`` branch.
"""

import numpy as np


def build_calo_root_map(sim_particle_ids_evt, ancestor_ids_evt,
                        hit_sim_ids, hit_crystal_ids, crystal_disk_map):
    """Build a mapping from (SimParticle, disk) -> calo-entrant root.

    The calo-entrant root for a SimParticle on a given disk is the highest
    ancestor that also deposited energy in that disk.  Cross-disk secondaries
    become their own calo-entrant.

    Parameters
    ----------
    sim_particle_ids_evt : array-like
        ``calomcsim.id`` for one event — SimParticle IDs present in calo.
    ancestor_ids_evt : list of array-like
        ``calomcsim.ancestorSimIds`` for one event — per-SimParticle
        ancestor chains (ordered child→root).
    hit_sim_ids : list of array-like
        ``calohitsmc.simParticleIds`` for one event — per-hit contributing
        SimParticle IDs.
    hit_crystal_ids : array-like
        ``calohits.crystalId_`` for one event.
    crystal_disk_map : dict
        crystalId -> diskId.

    Returns
    -------
    calo_root_map : dict
        ``{(simParticle_id, disk_id): calo_entrant_id}``
    """
    sim_ids_set = set(int(x) for x in sim_particle_ids_evt)

    # Determine which disks each SimParticle deposits in, from hit data.
    simp_disks = {}  # simP_id -> set of disk_ids
    for i in range(len(hit_sim_ids)):
        cryid = int(hit_crystal_ids[i])
        disk = crystal_disk_map.get(cryid, -1)
        if disk < 0:
            continue
        for pid in hit_sim_ids[i]:
            pid = int(pid)
            if pid not in simp_disks:
                simp_disks[pid] = set()
            simp_disks[pid].add(disk)

    # For each (SimParticle, disk), walk up ancestor chain to find
    # the highest ancestor that also deposited in the same disk.
    calo_root_map = {}
    for j in range(len(sim_particle_ids_evt)):
        sid = int(sim_particle_ids_evt[j])
        ancestors = [int(a) for a in ancestor_ids_evt[j]]
        for disk in simp_disks.get(sid, set()):
            calo_root = sid
            for a in ancestors:
                if a in sim_ids_set and disk in simp_disks.get(a, set()):
                    calo_root = a
            calo_root_map[(sid, disk)] = calo_root

    return calo_root_map


def assign_mc_truth_primary(sim_particle_ids, edeps, hit_disks,
                            edge_index, calo_root_map,
                            purity_threshold=0.7):
    """Assign edge labels using calo-entrant (primary-level) truth.

    Like :func:`assign_mc_truth` but groups energy deposits by calo-entrant
    root before computing purity.  This means that multiple SimParticle
    contributions from the same shower sum together, reducing ambiguity
    and collapsing secondary singletons into parent showers.

    Parameters
    ----------
    sim_particle_ids : list of lists
        Per-hit SimParticle IDs.
    edeps : list of lists
        Per-hit energy deposits (aligned with *sim_particle_ids*).
    hit_disks : np.ndarray of shape (n_hits,), int
        Disk ID per hit.
    edge_index : np.ndarray of shape (2, n_edges), int
        Graph edge index.
    calo_root_map : dict
        ``{(simParticle_id, disk_id): calo_entrant_id}`` from
        :func:`build_calo_root_map`.
    purity_threshold : float
        Minimum dominant calo-entrant purity to consider a hit
        non-ambiguous.  Default 0.7.

    Returns
    -------
    y_edge : np.ndarray of shape (n_edges,), int (0 or 1)
    edge_mask : np.ndarray of shape (n_edges,), bool (True = valid)
    hit_truth_cluster : np.ndarray of shape (n_hits,), int
        Truth-cluster ID per hit; -1 for ambiguous hits.
    is_ambiguous : np.ndarray of shape (n_hits,), bool
    """
    hit_disks = np.asarray(hit_disks)
    edge_index = np.asarray(edge_index)
    n_hits = len(sim_particle_ids)

    dominant_root = np.full(n_hits, -1, dtype=np.int64)
    is_ambiguous = np.ones(n_hits, dtype=bool)

    for i in range(n_hits):
        pids = sim_particle_ids[i]
        deps = edeps[i]
        if len(pids) == 0 or len(deps) == 0:
            continue
        deps_arr = np.asarray(deps, dtype=np.float64)
        total = deps_arr.sum()
        if total <= 0:
            continue

        disk = int(hit_disks[i])

        # Group energy deposits by calo-entrant root
        root_edep = {}
        for pid, dep in zip(pids, deps):
            root = calo_root_map.get((int(pid), disk), int(pid))
            root_edep[root] = root_edep.get(root, 0.0) + float(dep)

        best_root = max(root_edep, key=root_edep.get)
        purity = root_edep[best_root] / total
        if purity >= purity_threshold:
            is_ambiguous[i] = False
            dominant_root[i] = best_root

    # Build truth cluster IDs: unique (calo_root, disk) pairs
    hit_truth_cluster = np.full(n_hits, -1, dtype=np.int64)
    cluster_map = {}
    next_id = 0
    for i in range(n_hits):
        if is_ambiguous[i]:
            continue
        key = (int(dominant_root[i]), int(hit_disks[i]))
        if key not in cluster_map:
            cluster_map[key] = next_id
            next_id += 1
        hit_truth_cluster[i] = cluster_map[key]

    # Edge labels
    src = edge_index[0]
    dst = edge_index[1]
    amb_src = is_ambiguous[src]
    amb_dst = is_ambiguous[dst]
    edge_mask = ~(amb_src | amb_dst)

    tc_src = hit_truth_cluster[src]
    tc_dst = hit_truth_cluster[dst]
    y_edge = ((tc_src == tc_dst) & (tc_src != -1)).astype(np.int64)
    y_edge[~edge_mask] = 0

    return y_edge, edge_mask, hit_truth_cluster, is_ambiguous
