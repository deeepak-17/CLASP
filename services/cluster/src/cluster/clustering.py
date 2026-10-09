"""Clustering novelty (Week 10, P2): group clients by the *direction* of their
LoRA updates, instead of only by a hand-assigned project label.

Pipeline (each step is a function here):

    client adapters (B_i, A_i)  [+ the adapter each client started the round from]
        -> delta_W_i = B_i @ A_i per (layer, module)              adapter.delta_w
        -> update u_i = delta_W_i - delta_W_start_i               ``flatten_update``
        -> flatten and concatenate over (layer, module)           ``flatten_update``
        -> cosine-similarity matrix S_ij = <u_i,u_j>/(|u_i||u_j|) ``cosine_similarity_matrix``
        -> k-means (k=2) over the update directions               ``kmeans_on_similarity``
        -> cluster assignment, with warm-start comparison         ``ClusterAssigner``

Two equivalent routes to S are provided on purpose:

  * ``flatten_update`` materializes the literal flattened vector. Exact and
    simple, but a real 24-layer model has ~10^8 entries per client.
  * ``update_gram`` gets the same inner products *from the LoRA factors*
    (<B_i A_i, B_j A_j>_F = sum((B_i^T B_j) * (A_i A_j^T)), an r x r computation per
    module) without ever forming a full delta_W. ``cosine_similarity_matrix``
    uses this one; the tests assert both routes agree.

K-means runs on the unit-normalized update vectors. For unit vectors
|u_i - u_j|^2 = 2 - 2 S_ij, so Lloyd's algorithm can be carried out from S alone
(kernel form) with no vectors in memory; the tests cross-check the result
against scikit-learn's KMeans on the flattened vectors.

``ClusterAssigner`` is the policy layer: warm-start labels first, dynamic
re-clustering after round ``recluster_after_round`` (D4), and a *static
fallback* to the warm-start assignment whenever the dynamic result cannot be
trusted (too few clients, degenerate/tiny clusters, weak separation, or an
error). It never invents an assignment for a client it has not seen.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from cluster.adapter_format import LoRAAdapter

_MAX_LABEL_PERMUTATION_K = 8


# --------------------------------------------------------------------------
# update representation: delta_W reconstruction -> flatten
# --------------------------------------------------------------------------


def flatten_delta(adapter: LoRAAdapter) -> np.ndarray:
    """Concatenate delta_W = B @ A over (layer, module) into one 1-D float64 vector."""
    adapter.validate()
    return np.concatenate(
        [
            adapter.delta_w(m, layer).astype(np.float64).ravel()
            for layer in adapter.layer_indices
            for m in adapter.target_modules
        ]
    )


def flatten_update(adapter: LoRAAdapter, start: LoRAAdapter | None = None) -> np.ndarray:
    """Flattened *update*: flatten(delta_W(adapter)) - flatten(delta_W(start)).

    ``start`` is the adapter the client received at the beginning of the round;
    ``None`` means a zero start (e.g. the PEFT B=0 init).
    """
    vec = flatten_delta(adapter)
    if start is None:
        return vec
    start_vec = flatten_delta(start)
    if start_vec.shape != vec.shape:
        raise ValueError(
            f"adapter and start adapter differ in shape: {vec.shape} vs {start_vec.shape}"
        )
    return vec - start_vec


def _check_compatible(a: LoRAAdapter, b: LoRAAdapter) -> None:
    if (
        a.target_modules != b.target_modules
        or a.num_layers != b.num_layers
        or a.rank != b.rank
    ):
        raise ValueError(
            "adapters must share rank/target_modules/num_layers to be compared "
            f"(got r={a.rank}/{b.rank}, modules={a.target_modules}/{b.target_modules}, "
            f"layers={a.num_layers}/{b.num_layers})"
        )


def delta_gram(adapters: Sequence[LoRAAdapter]) -> np.ndarray:
    """G_ij = <delta_W(adapter_i), delta_W(adapter_j)>_F summed over (layer, module),
    computed from the LoRA factors (no full delta_W is formed)."""
    n = len(adapters)
    if n == 0:
        raise ValueError("no adapters")
    for ad in adapters:
        ad.validate()
        _check_compatible(adapters[0], ad)
    gram = np.zeros((n, n), dtype=np.float64)
    ref = adapters[0]
    for layer in ref.layer_indices:
        for module in ref.target_modules:
            bs = [ad.modules[layer][module]["lora_B"].astype(np.float64) for ad in adapters]
            as_ = [ad.modules[layer][module]["lora_A"].astype(np.float64) for ad in adapters]
            for i in range(n):
                for j in range(i, n):
                    # tr(A_i^T B_i^T B_j A_j) = sum((B_i^T B_j) * (A_i A_j^T))
                    val = float(np.sum((bs[i].T @ bs[j]) * (as_[i] @ as_[j].T)))
                    gram[i, j] += val
                    if j != i:
                        gram[j, i] += val
    return gram


def update_gram(
    adapters: Sequence[LoRAAdapter],
    starts: Sequence[LoRAAdapter | None] | None = None,
) -> np.ndarray:
    """Gram matrix of the *updates* u_i = delta_W(adapter_i) - delta_W(start_i).

    Expands <u_i,u_j> = G_ij - G_i,s_j - G_s_i,j + G_s_i,s_j over one shared
    Gram of [adapters..., distinct starts...]; ``None`` start = zero start.
    """
    n = len(adapters)
    starts = list(starts) if starts is not None else [None] * n
    if len(starts) != n:
        raise ValueError(f"starts has {len(starts)} entries for {n} adapters")
    distinct: list[LoRAAdapter] = []
    start_slot: list[int | None] = []
    for st in starts:
        if st is None:
            start_slot.append(None)
            continue
        for k, known in enumerate(distinct):
            if known is st:
                start_slot.append(k)
                break
        else:
            distinct.append(st)
            start_slot.append(len(distinct) - 1)
    gram = delta_gram([*adapters, *distinct])

    def g(i: int | None, j: int | None) -> float:
        # index None = the zero start, whose inner product with anything is 0
        if i is None or j is None:
            return 0.0
        return gram[i, j]

    def idx(slot: int | None) -> int | None:
        return None if slot is None else n + slot

    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        si = idx(start_slot[i])
        for j in range(n):
            sj = idx(start_slot[j])
            out[i, j] = g(i, j) - g(i, sj) - g(si, j) + g(si, sj)
    return (out + out.T) / 2.0


def cosine_similarity_matrix(
    adapters: Sequence[LoRAAdapter],
    starts: Sequence[LoRAAdapter | None] | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """(n, n) cosine similarity of the clients' flattened updates, in [-1, 1].

    A client whose update is (numerically) zero has no direction: its
    similarity to every other client is 0 and its diagonal entry is 1.
    """
    gram = update_gram(adapters, starts)
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    scale = np.where(norms > eps, norms, 1.0)
    sim = gram / np.outer(scale, scale)
    zero = norms <= eps
    sim[zero, :] = 0.0
    sim[:, zero] = 0.0
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, -1.0, 1.0)


def cosine_similarity_from_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Reference cosine matrix straight from flattened vectors (used by tests)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    scale = np.where(norms > eps, norms, 1.0)
    unit = vectors / scale[:, None]
    sim = unit @ unit.T
    zero = norms <= eps
    sim[zero, :] = 0.0
    sim[:, zero] = 0.0
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, -1.0, 1.0)


# --------------------------------------------------------------------------
# k-means on update directions
# --------------------------------------------------------------------------


def _kernel_distances(sim: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Squared distance of every unit-vector point to each cluster centroid, from S."""
    n = sim.shape[0]
    dist = np.full((n, k), np.inf)
    for c in range(k):
        members = labels == c
        m = int(members.sum())
        if m == 0:
            continue
        within = sim[np.ix_(members, members)].sum() / (m * m)
        cross = sim[:, members].sum(axis=1) / m
        dist[:, c] = 1.0 + within - 2.0 * cross
    return dist


def _init_labels(sim: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding on d^2 = 2 - 2 S, then nearest-seed labels."""
    n = sim.shape[0]
    d2 = np.clip(2.0 - 2.0 * sim, 0.0, None)
    seeds = [int(rng.integers(n))]
    while len(seeds) < k:
        nearest = d2[:, seeds].min(axis=1)
        total = nearest.sum()
        if total <= 0:  # all remaining points coincide with a seed
            remaining = [i for i in range(n) if i not in seeds]
            seeds.append(int(rng.choice(remaining)))
            continue
        seeds.append(int(rng.choice(n, p=nearest / total)))
    return np.argmin(d2[:, seeds], axis=1)


def kmeans_on_similarity(
    sim: np.ndarray,
    k: int = 2,
    seed: int = 0,
    n_init: int = 10,
    max_iter: int = 100,
) -> tuple[np.ndarray, float]:
    """K-means (Lloyd) on the unit update vectors described by cosine matrix ``sim``.

    Returns ``(labels, inertia)``, the best of ``n_init`` seeded restarts.
    Labels are canonicalized so cluster ids appear in order of first use
    (point 0 is always in cluster 0), which makes results seed-order stable.
    """
    sim = np.asarray(sim, dtype=np.float64)
    n = sim.shape[0]
    if sim.shape != (n, n):
        raise ValueError("sim must be square")
    if k < 1 or k > n:
        raise ValueError(f"need 1 <= k <= n_points (k={k}, n={n})")
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_inertia = np.inf
    for _ in range(n_init):
        labels = _init_labels(sim, k, rng)
        for _ in range(max_iter):
            dist = _kernel_distances(sim, labels, k)
            new = np.argmin(dist, axis=1)
            # an emptied cluster is re-seeded with the point farthest from its centroid
            for c in range(k):
                if not np.any(new == c):
                    own = dist[np.arange(n), new]
                    new[int(np.argmax(own))] = c
            if np.array_equal(new, labels):
                break
            labels = new
        dist = _kernel_distances(sim, labels, k)
        inertia = float(dist[np.arange(n), labels].sum())
        if inertia < best_inertia - 1e-12:
            best_inertia, best_labels = inertia, labels
    assert best_labels is not None
    return _canonical(best_labels), best_inertia


def _canonical(labels: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    out = np.empty_like(labels)
    for i, lab in enumerate(labels):
        mapping.setdefault(int(lab), len(mapping))
        out[i] = mapping[int(lab)]
    return out


# --------------------------------------------------------------------------
# assignment quality + comparison utilities
# --------------------------------------------------------------------------


def cohesion_and_separation(sim: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(mean within-cluster off-diagonal similarity, mean between-cluster similarity).

    Within is taken over pairs of *distinct* members; a singleton cluster
    contributes no pairs. Either value is ``nan`` when no such pair exists.
    """
    labels = np.asarray(labels)
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(len(labels), dtype=bool)
    within = sim[same & off_diag]
    between = sim[~same]
    return (
        float(within.mean()) if within.size else float("nan"),
        float(between.mean()) if between.size else float("nan"),
    )


def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> float:
    """Adjusted Rand index of two labelings (1.0 = identical partitions)."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n != len(b):
        raise ValueError("labelings differ in length")
    if n < 2:
        return 1.0

    def comb2(x):
        return x * (x - 1) / 2.0

    ca, cb = np.unique(a), np.unique(b)
    table = np.array([[np.sum((a == i) & (b == j)) for j in cb] for i in ca], dtype=np.float64)
    sum_ij = comb2(table).sum()
    sum_a = comb2(table.sum(axis=1)).sum()
    sum_b = comb2(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(n)
    max_index = (sum_a + sum_b) / 2.0
    if max_index == expected:
        return 1.0
    return float((sum_ij - expected) / (max_index - expected))


def align_labels(
    labels: Sequence[int], reference: Sequence[int], k: int
) -> tuple[np.ndarray, dict[int, int]]:
    """Relabel ``labels`` (values in range(k)) to best agree with ``reference``.

    k-means label ids are arbitrary; this finds the permutation of ids that
    maximizes agreement with an existing assignment so "cluster 0" keeps
    meaning the same cluster across rounds.
    """
    if k > _MAX_LABEL_PERMUTATION_K:
        raise ValueError(f"k={k} too large for exhaustive alignment (max {_MAX_LABEL_PERMUTATION_K})")
    labels, reference = np.asarray(labels), np.asarray(reference)
    best_perm: tuple[int, ...] = tuple(range(k))
    best_agree = -1
    for perm in itertools.permutations(range(k)):
        agree = int(np.sum(np.asarray(perm)[labels] == reference))
        if agree > best_agree:
            best_agree, best_perm = agree, perm
    mapping = {i: best_perm[i] for i in range(k)}
    return np.asarray(best_perm)[labels], mapping


# --------------------------------------------------------------------------
# policy: warm start -> dynamic re-clustering -> static fallback
# --------------------------------------------------------------------------


@dataclass
class AssignmentRecord:
    """One (re-)assignment decision, kept for logs / the Phase-II report."""

    completed_rounds: int
    method: str  # "warm_start" | "dynamic" | "static_fallback" | "static"
    assignment: dict[str, str]
    reason: str = ""
    moved: dict[str, tuple[str, str]] = field(default_factory=dict)  # client -> (old, new)
    clients_clustered: list[str] = field(default_factory=list)
    similarity: list[list[float]] | None = None
    cohesion: float | None = None
    separation: float | None = None
    comparison: dict[str, float] | None = None  # dynamic vs warm-start


class ClusterAssigner:
    """Assigns clients to clusters: warm-start labels, then dynamic re-clustering.

    ``warm_start`` maps client_id -> cluster_id (e.g. project type: "web" vs
    "scientific"). It is the assignment used until re-clustering is due, and the
    static fallback afterwards. ``cluster_ids`` fixes the id set (and so ``k``).

    ``mode="static"`` never re-clusters (the D12 fallback path, used as the
    baseline in the dynamic-vs-static ablation).
    """

    def __init__(
        self,
        warm_start: Mapping[str, str],
        cluster_ids: Sequence[str] | None = None,
        *,
        k: int | None = None,
        recluster_after_round: int = 2,
        recluster_every: int | None = None,
        mode: str = "dynamic",
        min_cluster_size: int = 1,
        min_separation: float = 0.1,
        seed: int = 0,
    ) -> None:
        if mode not in ("dynamic", "static"):
            raise ValueError(f"unknown mode {mode!r}")
        self.cluster_ids: tuple[str, ...] = tuple(
            cluster_ids if cluster_ids is not None else sorted(set(warm_start.values()))
        )
        if not set(warm_start.values()) <= set(self.cluster_ids):
            raise ValueError("warm_start references a cluster id outside cluster_ids")
        self.k = k if k is not None else len(self.cluster_ids)
        if self.k != len(self.cluster_ids):
            raise ValueError("k must equal the number of cluster_ids")
        self.warm_start: dict[str, str] = dict(warm_start)
        self.assignment: dict[str, str] = dict(warm_start)
        self.recluster_after_round = recluster_after_round
        self.recluster_every = recluster_every
        self.mode = mode
        self.min_cluster_size = min_cluster_size
        self.min_separation = min_separation
        self.seed = seed
        self.history: list[AssignmentRecord] = [
            AssignmentRecord(0, "warm_start", dict(self.assignment), "initial warm-start labels")
        ]

    # ---- queries --------------------------------------------------------

    def members(self, cluster_id: str) -> list[str]:
        return sorted(c for c, cl in self.assignment.items() if cl == cluster_id)

    def due(self, completed_rounds: int) -> bool:
        if self.mode == "static" or completed_rounds < self.recluster_after_round:
            return False
        if completed_rounds == self.recluster_after_round:
            return True
        if self.recluster_every:
            return (completed_rounds - self.recluster_after_round) % self.recluster_every == 0
        return False

    # ---- re-clustering --------------------------------------------------

    def maybe_recluster(
        self,
        completed_rounds: int,
        updates: Mapping[str, LoRAAdapter],
        starts: Mapping[str, LoRAAdapter | None] | None = None,
    ) -> AssignmentRecord | None:
        """Re-cluster if due. ``updates``: client_id -> adapter it returned this
        round; ``starts``: client_id -> adapter it started from. Clients missing
        from ``updates`` (dropouts) keep their current assignment. Returns the
        record, or ``None`` when nothing was due."""
        if not self.due(completed_rounds):
            return None
        return self.recluster(completed_rounds, updates, starts)

    def recluster(
        self,
        completed_rounds: int,
        updates: Mapping[str, LoRAAdapter],
        starts: Mapping[str, LoRAAdapter | None] | None = None,
    ) -> AssignmentRecord:
        starts = starts or {}
        present = sorted(c for c in updates if c in self.assignment)
        unknown = sorted(set(updates) - set(self.assignment))

        def fallback(reason: str, **extra) -> AssignmentRecord:
            rec = AssignmentRecord(
                completed_rounds,
                "static_fallback",
                dict(self.assignment),
                reason,
                clients_clustered=present,
                **extra,
            )
            self.history.append(rec)
            return rec

        if unknown:
            return fallback(f"updates from unassigned client(s) {unknown}; refusing to guess")
        if len(present) < self.k:
            return fallback(f"only {len(present)} client update(s) present for k={self.k}")
        try:
            adapters = [updates[c] for c in present]
            sim = cosine_similarity_matrix(adapters, [starts.get(c) for c in present])
            labels, _ = kmeans_on_similarity(sim, self.k, seed=self.seed)
        except Exception as exc:  # noqa: BLE001 - any failure => safe static assignment
            return fallback(f"clustering failed: {type(exc).__name__}: {exc}")

        sim_list = sim.tolist()
        sizes = np.bincount(labels, minlength=self.k)
        within, between = cohesion_and_separation(sim, labels)
        gap = (within if not np.isnan(within) else 0.0) - (
            between if not np.isnan(between) else 0.0
        )
        if int(sizes.min()) < self.min_cluster_size:
            return fallback(
                f"degenerate clustering (cluster sizes {sizes.tolist()} < min {self.min_cluster_size})",
                similarity=sim_list,
                cohesion=None if np.isnan(within) else within,
                separation=None if np.isnan(between) else between,
            )
        if gap < self.min_separation:
            return fallback(
                f"weak separation (within-between gap {gap:.3f} < {self.min_separation})",
                similarity=sim_list,
                cohesion=None if np.isnan(within) else within,
                separation=None if np.isnan(between) else between,
            )

        # keep cluster ids stable: map k-means ids onto the existing assignment
        index_of = {cid: i for i, cid in enumerate(self.cluster_ids)}
        current = [index_of[self.assignment[c]] for c in present]
        aligned, _ = align_labels(labels, current, self.k)
        new_assignment = dict(self.assignment)
        for client, lab in zip(present, aligned):
            new_assignment[client] = self.cluster_ids[int(lab)]
        moved = {
            c: (self.assignment[c], new_assignment[c])
            for c in present
            if new_assignment[c] != self.assignment[c]
        }

        warm = [index_of[self.warm_start[c]] for c in present if c in self.warm_start]
        comparison = None
        if len(warm) == len(present):
            comparison = self.compare_to_warm_start(sim, aligned, warm)

        rec = AssignmentRecord(
            completed_rounds,
            "dynamic",
            dict(new_assignment),
            "k-means on cosine similarity of flattened delta_W updates",
            moved=moved,
            clients_clustered=present,
            similarity=sim_list,
            cohesion=None if np.isnan(within) else within,
            separation=None if np.isnan(between) else between,
            comparison=comparison,
        )
        self.assignment = new_assignment
        self.history.append(rec)
        return rec

    @staticmethod
    def compare_to_warm_start(
        sim: np.ndarray, dynamic: Sequence[int], warm: Sequence[int]
    ) -> dict[str, float]:
        """Dynamic vs warm-start assignment on the same clients (W10 Thu).

        ``agreement``  fraction of clients given the same cluster id;
        ``ari``        adjusted Rand index of the two partitions;
        ``*_cohesion`` / ``*_separation``  mean within/between-cluster cosine of
        each partition, i.e. how well each one groups update directions.
        """
        dynamic, warm = np.asarray(dynamic), np.asarray(warm)
        d_in, d_out = cohesion_and_separation(sim, dynamic)
        w_in, w_out = cohesion_and_separation(sim, warm)
        return {
            "agreement": float(np.mean(dynamic == warm)),
            "ari": adjusted_rand_index(dynamic, warm),
            "dynamic_cohesion": d_in,
            "dynamic_separation": d_out,
            "warm_start_cohesion": w_in,
            "warm_start_separation": w_out,
        }
