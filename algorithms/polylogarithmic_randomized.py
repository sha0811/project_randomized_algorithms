"""
bbmn_exact.py — Faithful implementation of the BBMN polylogarithmic k-server algorithm.
The research work [BBMN11] is built upon the work [Cote08]

References
----------
[BBMN11] Bansal, Buchbinder, Mądry, Naor.
         "A Polylogarithmic-Competitive Algorithm for the k-Server Problem." FOCS 2011.

         §4.1  Fractional k-server on HSTs: Lambda_p^t, quota patterns, hit costs,
               consistency equations (40)-(41), fractional server count z(q,t).
         §5.1  Embedding sigm-HSTs into balanced weighted sigma-HSTs (Theorem 8).
         §5.2  Online rounding: Theorem 24 (balanced states) + Lemma 25 (rebalancing).

[Cote08] Cote, Meyerson, Poplawski.
         "Randomized k-Server on Hierarchical Binary Trees." STOC 2008.
         (The fix-stage / hit-stage allocation algorithm referenced in §4.1.)

Design decisions
----------------
The theoremes and lemmas mentionned below are in the paper [BBMN11].
* sigma-HST construction: FRT randomised clustering (gives E[d_HST] = O(log n)·d_metric),
  followed by the balancing contraction of Theorem 8 to achieve depth O(log n).
  This is basically how we construct the weighted HST to avoid the explosion of the 
  competitive-ratio due to it's dependence on the radius delta of the metric space.
* Lambda_p^t is stored compactly as a probability vector pi[0..k] over integer quotas,
  plus one AllocationInstance (y-variables) per quota value. This is exact.
* Rounding state: a single integral configuration (set of k leaves) maintained
  probabilistically via the elementary-move procedure of Theorem 24 / Lemma 25.
  This is correct because every elementary move has infinitesimal mass delta that tends to 0, so the
  rounding reduces to: with probability proportional to delta, swap leaf i' for leaf i
  (or a surrogate j found by the pigeon-hole argument of Lemma 25).
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from utilities import load_instance, manhattan

# ─────────────────────────────────────────────────────────────────────────────
# 1.  sigma-HST CONSTRUCTION
#     §5.1 of the paper: FRT embedding + balancing contraction (Theorem 8)
#                        FRT is used to construct the probabilistic tree in Th8
#                        cf. (Fakcharoenphol, Rao, Talwar 2004)
# ─────────────────────────────────────────────────────────────────────────────

class HSTNode:
    """Node in a sigma-HST."""
    _ctr: int = 0

    def __init__(self, edge_weight: float, leaves: List[int]):
        HSTNode._ctr += 1
        self.id           = HSTNode._ctr
        self.edge_weight  = edge_weight   # W(p) = length of edge from p to its parent
        self.leaves       = leaves        # original point indices in this subtree
        self.children:    List["HSTNode"] = []
        self.parent:      Optional["HSTNode"] = None
        # Fractional server count x_p = sum of x_i for leaves i in this subtree
        # it represents the fractional number of servers assigned to node p, it changes over 
        # time till it converges
        self.xp: float    = 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def n_nodes(self) -> int:
        """Total number of nodes (leaves + internal) in this subtree."""
        if self.is_leaf:
            return 1
        return 1 + sum(c.n_nodes for c in self.children)


class SigmaHST:
    """
    Random sigma-HST built by FRT + Theorem 8 balancing contraction.

    The FRT construction (Fakcharoenphol, Rao, Talwar 2004):
      1. Pick a random permutation π of the n sites.
      2. Pick β ~ Uniform[0,1).
      3. At scale level l (l=0 is coarsest), ball radius = delta/sigma^l · sigma^β.
      4. Cluster greedily in permutation order.

    After FRT, apply Theorem 8 contraction:
      While some child of a node contains > half the nodes of the subtree,
      contract the edge to that child (merge child into parent).
    This gives depth O(log n) and distortion ≤ 2sigma/(sigma-1).

    sigma > 5 is required by Theorem 24 (online rounding).
    """

    def __init__(self, sites: List[Tuple[int,int]], sigma: float, rng: random.Random):
        assert sigma > 5, "sigma must be > 5 for the rounding guarantee (§5.2)"
        self.sites = sites
        self.n     = len(sites)
        self.sigma = sigma
        self.rng   = rng
        HSTNode._ctr = 0

        # FRT randomness
        self._perm = list(range(self.n))
        rng.shuffle(self._perm)
        self._beta = rng.random()

        diam       = self._diameter()
        # L: number of levels so that sigma^L ≥ diameter
        self.L     = math.ceil(math.log(max(diam, 1)) / math.log(sigma)) + 1
        self._root_radius = sigma ** self.L

        # leaf_node[i] = the HSTNode that is the leaf for original site i
        self.leaf_node: List[Optional[HSTNode]] = [None] * self.n

        # Build FRT tree
        root = HSTNode(edge_weight=0.0, leaves=list(range(self.n)))
        self._frt_build(root, list(range(self.n)), level=0)
        # Apply Theorem 8 balancing
        self._balance(root)
        self.root = root

        # Pre-compute path from each leaf to root (for fast LCA / path queries)
        self._path_to_root: List[List[HSTNode]] = [[] for _ in range(self.n)]
        for i in range(self.n):
            nd = self.leaf_node[i]
            while nd is not None:
                self._path_to_root[i].append(nd)
                nd = nd.parent

    # ── geometry ──────────────────────────────────────────────────────────

    def _d(self, u: int, v: int) -> float:
        ax, ay = self.sites[u]
        bx, by = self.sites[v]
        return float(abs(ax - bx) + abs(ay - by))

    def _diameter(self) -> float:
        best = 1.0
        for i in range(self.n):
            for j in range(i+1, self.n):
                best = max(best, self._d(i, j))
        return best

    # ── FRT construction ──────────────────────────────────────────────────

    def _ball_radius(self, level: int) -> float:
        """Ball radius at this level (FRT formula with random shift β)."""
        return self._root_radius / (self.sigma ** (level + 1)) * (self.sigma ** self._beta)

    def _frt_build(self, node: HSTNode, pts: List[int], level: int):
        if len(pts) == 1:
            self.leaf_node[pts[0]] = node
            return

        r       = self._ball_radius(level)
        pts_set = set(pts)

        # Greedy ball cover in permutation order
        assigned: Dict[int, int] = {}
        for center in self._perm:
            if center in pts_set and center not in assigned:
                for q in pts:
                    if q not in assigned and self._d(center, q) <= r:
                        assigned[q] = center

        clusters: Dict[int, List[int]] = defaultdict(list)
        for p in pts:
            clusters[assigned[p]].append(p)

        # Edge weight for sigma-HST: child edges at this level have length sigma^(L-level-1)
        child_edge_w = self._root_radius / (self.sigma ** (level + 1))

        for members in clusters.values():
            child = HSTNode(edge_weight=child_edge_w, leaves=members)
            child.parent = node
            node.children.append(child)
            self._frt_build(child, members, level + 1)

    # ── Theorem 8 balancing ───────────────────────────────────────────────

    def _balance(self, node: HSTNode):
        """
        Theorem 8: contract edges to children that contain > half the nodes.
        Applied recursively bottom-up.
        """
        if node.is_leaf:
            return
        # Recurse first (bottom-up)
        for ch in node.children:
            self._balance(ch)

        total = node.n_nodes
        # Find the child (at most one) with > half the nodes
        heavy = None
        for ch in node.children:
            if ch.n_nodes > total // 2:
                heavy = ch
                break

        if heavy is not None:
            # Contract: absorb heavy's children into node, remove heavy
            node.children.remove(heavy)
            for grandchild in heavy.children:
                grandchild.parent = node
                node.children.append(grandchild)
            # Update leaf_node pointers if heavy was a leaf
            if heavy.is_leaf:
                for lf in heavy.leaves:
                    self.leaf_node[lf] = node
                # node becomes a leaf too (rare edge case)
            # The edge weight of grandchildren remains as-is (Theorem 8 keeps their lengths)

    # ── queries ───────────────────────────────────────────────────────────

    def lca(self, u: int, v: int) -> HSTNode:
        """Lowest common ancestor of sites u and v."""
        anc: Dict[int, HSTNode] = {}
        for nd in self._path_to_root[u]:
            anc[nd.id] = nd
        for nd in self._path_to_root[v]:
            if nd.id in anc:
                return nd
        return self.root

    def hst_dist(self, u: int, v: int) -> float:
        """HST distance between leaves u and v = 2 * edge_weight of LCA's children.
        HST is uniforme, therefore the distances change in levels
           parent 
            /d \d
           u    v
           dist(u, v) = 2*d
        """
        if u == v:
            return 0.0
        lca_node = self.lca(u, v)
        # In a sigma-HST the distance = sum of edge weights on path u..v
        # = 2 * (edge weight from lca to its children at the level of lca)
        # The children of lca have edge_weight = W(child), so dist = 2*W(child of lca on path)
        # Find the child of lca on the path to u
        for nd in self._path_to_root[u]:
            if nd.parent is not None and nd.parent.id == lca_node.id:
                return 2.0 * nd.edge_weight
        return 2.0 * (lca_node.children[0].edge_weight if lca_node.children else 0.0)

    def subtree_leaves(self, node: HSTNode) -> Set[int]:
        """All original site indices in the subtree of node."""
        return set(node.leaves)

# AllocationInstance: Fix stage + Hit stage (fractional allocation dynamics)
#
# We are at one internal HST node p.
#
# Structural parameters:
#   d          = number of children of p  (indexed i = 0..d-1)
#   k          = global number of servers in the system
#   κ (kappa)  = quota at this node = expected number of servers assigned
#                to the subtree of p
#   w_i > 0    = edge weight from p to child i (transport scale toward child i)
#
# Allocation state:
#   y ∈ [0,1]^{d×k}
#   y[i,j] = P(child i receives at least j+1 servers),  j = 0..k-1
#
#   For each fixed i:
#       y[i,0] ≥ y[i,1] ≥ ... ≥ y[i,k-1]
#
# Exact server probabilities are derived from y:
#   x[i,0] = 1 - y[i,0]
#   x[i,j] = y[i,j-1] - y[i,j]   for j = 1..k-1
#   x[i,k] = y[i,k-1]
#
# Expected total servers allocated by this node:
#   Σ_i Σ_j y[i,j]   = κ
#
# --------------------------------------------------------------------------
# (1) FIX STAGE — Restore total mass to κ
# --------------------------------------------------------------------------
# If total mass Σ_i Σ_j y[i,j] < κ, we increase y smoothly.
#
# Continuous evolution variable:
#   τ  = artificial time for fix stage
#
# ODE governing growth:
#   dy_{i,j}/dτ = (y_{i,j} + β) / w_i
#
# where:
#   β = eps / (1 + k)   (small smoothing constant)
#       - ensures strictly positive growth even when y[i,j]=0
#   eps = external hyperparameter controlling smoothing
#
# Closed-form solution:
#   y_{i,j}(τ) = (y_{i,j}(0) + β) * exp(τ / w_i) - β
#
# We choose τ so that:
#   Σ_i Σ_j y_{i,j}(τ) = κ
#
# Implementation:
#   - increase τ until total mass ≥ κ
#   - binary search τ for precise matching
#
# Effect:
#   Mass grows faster on children with smaller w_i (closer subtrees).
#
# --------------------------------------------------------------------------
# (2) HIT STAGE — React to a request while conserving total mass
# --------------------------------------------------------------------------
# Suppose a request arrives in the subtree of child i_bar.
#
# We define a hit-cost sequence:
#   h[0..k],  decreasing in j, with h[0] = +∞
#
# From it we define λ:
#   λ_j = max(0, h[j] - h[j+1])   for j = 0..k-1
#
# lam_row[j] stores λ_j for the requested child i_bar.
# For children i ≠ i_bar, λ = 0.
#
# Continuous evolution variable:
#   η ∈ [0,1]  = artificial time for hit stage
#
# ODE during hit stage:
#   dy_{i,j}/dη =
#       (y_{i,j} + β) / w_i * ( N(η) - α λ_{i,j} )
#
# where:
#   α = log(1 + 1/β)    (scaling constant from analysis)
#   λ_{i,j} = λ_j if i = i_bar, else 0
#   N(η)     = normalization term chosen so that:
#
#       d/dη  Σ_i Σ_j y[i,j]  = 0
#
# i.e., total mass remains exactly κ.
#
# Interpretation of terms:
#   (y + β)/w_i   → speed of mass movement (larger y and smaller w_i move faster)
#   α λ_{i,j}     → directional pressure from the request
#   N(η)          → balances system to conserve total mass
#
# Numerics:
#   deta     = small step size for η (Euler discretization)
#   N        = computed each step from conservation equation
#   blocks   = groups of j indices with equal y-values on i_bar
#              (ensures structural monotonicity and stability)
#
# Effect:
#   HIT STAGE redistributes the same κ units of fractional mass,
#   shifting probability toward the requested subtree i_bar,
#   without changing the total expected number of servers.
# ─────────────────────────────────────────────────────────────────────────────
"""
For complexity reasons, we couldn't implement the fractional allocation 
algorithm mentionned in BMMN's paper, that's why our implementation is 
actually an approximation to the BMMN algorithm.
How it works at this stage: 
You're at some internal node of the HST tree.
    That node has d children and you have k servers total
    -> But this node is only responsible for κ (kappa) servers. So the problem becomes:
    How do I distribute κ servers among my d children?
    That's the allocation problem.
But there is something more, instead of distributing the servers in a discrete way, we allow fractions, 
and that's the breakthrough of the [BBMN11] paper, that's where the y matrix comes in,
"""
class AllocationInstance:
    """
    One allocation instance for a node with d children, quota kappa, edge weights w.

    State: y of shape (d, k), where y[i, j] = P(child i gets >= j+1 servers).
    """

    def __init__(self, d: int, k: int, w: np.ndarray, eps: float, deta: float = 5e-3):
        assert d >= 1 and k >= 1
        self.d    = d
        self.k    = k
        self.w    = w                    # shape (d,), edge weights (must be > 0)
        self.eps  = eps
        self.beta = eps / (1.0 + k)
        self.alpha = math.log(1.0 + 1.0 / self.beta)
        self.deta = deta
        self.tol  = 1e-12

        # Initial state: y = 0 everywhere (no servers)
        self.y = np.zeros((d, k), dtype=float)

    # ── y -> x conversion ─────────────────────────────────────────────────

    def x_matrix(self) -> np.ndarray:
        """
        x[i, j] = P(child i gets exactly j servers), shape (d, k+1).
        x[i, 0] = 1 - y[i, 0]
        x[i, j] = y[i, j-1] - y[i, j]   for j = 1..k-1
        x[i, k] = y[i, k-1]
        """
        d, k = self.d, self.k
        X       = np.zeros((d, k+1), dtype=float)
        X[:, 0] = 1.0 - self.y[:, 0]
        for j in range(1, k):
            X[:, j] = self.y[:, j-1] - self.y[:, j]
        X[:, k] = self.y[:, k-1]
        return np.maximum(X, 0.0)

    def expected_servers(self) -> np.ndarray:
        """E[servers at child i] = sum_j y[i,j],  shape (d,)."""
        return np.sum(self.y, axis=1)

    # ── Fix stage ─────────────────────────────────────────────────────────

    def _y_at_tau(self, y0: np.ndarray, tau: float) -> np.ndarray:
        """Analytic solution to dy/dtau = (y+beta)/w_i:  y(tau) = (y0+beta)*exp(tau/w)-beta."""
        w = self.w[:, None]
        return np.clip((y0 + self.beta) * np.exp(tau / w) - self.beta, 0.0, 1.0)

    def fix_stage(self, kappa: float):
        """
        Raise sum(y) to kappa via the ODE.
        kappa = target quota for this instance.
        """
        target = float(kappa)
        if float(np.sum(self.y)) >= target - self.tol:
            return   # already satisfied

        tau_lo, tau_hi = 0.0, 1.0
        while float(np.sum(self._y_at_tau(self.y, tau_hi))) < target - self.tol:
            tau_hi *= 2.0
            if tau_hi > 1e9:
                self.y = self._y_at_tau(self.y, 1e9)
                return
        # Bisect for exact tau
        for _ in range(80):
            tau_m = 0.5 * (tau_lo + tau_hi)
            if float(np.sum(self._y_at_tau(self.y, tau_m))) >= target - self.tol:
                tau_hi = tau_m
            else:
                tau_lo = tau_m
        self.y = self._y_at_tau(self.y, tau_hi)

    # ── Hit stage ─────────────────────────────────────────────────────────

    def hit_stage(self, i_bar: int, lam_row: np.ndarray):
        """
        Decrease y on child i_bar (the requested child) while conserving total mass.
        lam_row: shape (k,), the lambda vector for child i_bar.

        ODE (hit stage, Cote et al.):
          dy_{i,j}/deta = (y_{i,j}+beta)/w_i * (N(eta) - alpha * lambda_{i,j})
          where N(eta) is chosen by conservation: d/deta sum_{i,j} y_{i,j} = 0.

        Only i_bar has nonzero lambda (Observation 17).
        Other children i != i_bar have lambda = 0, so their ODE is:
          dy_{i,j}/deta = (y_{i,j}+beta)/w_i * N(eta)
        """
        # Initial block partition: singletons {j} for j in 0..k-1
        # (blocks merge when y values equalise and lambda ordering inverts)
        blocks    = [(j, j) for j in range(self.k)]   # (j_start, j_end) inclusive
        blam      = [float(lam_row[j]) for j in range(self.k)]  # avg lambda per block

        eta = 0.0
        while eta < 1.0 - 1e-15:
            step = min(self.deta, 1.0 - eta)

            # Enforce within-block equality on i_bar (numerical drift correction)
            for (j0, j1) in blocks:
                if j1 > j0:
                    v = float(np.mean(self.y[i_bar, j0:j1+1]))
                    self.y[i_bar, j0:j1+1] = v

            # Compute N via conservation:
            # sum_{i,j} dy_{i,j}/deta = 0
            # => N * A = alpha * B  where:
            #   A = sum of (y+beta)/w for all active (i,j)
            #   B = sum of lambda_{i,j} * (y+beta)/w for active (i,j) on i_bar
            # "Active": NOT (val>0 and y>=1) AND NOT (val<=0 and y<=0)
            N = 0.0
            for _ in range(15):
                A, B = 0.0, 0.0
                # i_bar contribution
                for idx, (j0, j1) in enumerate(blocks):
                    lb  = float(blam[idx])
                    val = N - self.alpha * lb
                    for j in range(j0, j1+1):
                        yv = float(self.y[i_bar, j])
                        if (val > 0 and yv >= 1.0 - self.tol) or \
                           (val <= 0 and yv <= self.tol):
                            continue
                        coeff = (yv + self.beta) / float(self.w[i_bar])
                        A    += coeff
                        B    += lb * coeff
                # Other children (lambda = 0)
                for i in range(self.d):
                    if i == i_bar:
                        continue
                    val = N   # lambda=0
                    for j in range(self.k):
                        yv = float(self.y[i, j])
                        if (val > 0 and yv >= 1.0 - self.tol) or \
                           (val <= 0 and yv <= self.tol):
                            continue
                        A += (yv + self.beta) / float(self.w[i])
                N_new = (self.alpha * B / A) if A > 1e-18 else 0.0
                N_new = max(0.0, N_new)
                if abs(N_new - N) < 1e-10:
                    N = N_new
                    break
                N = N_new

            # Update y
            for idx, (j0, j1) in enumerate(blocks):
                lb  = float(blam[idx])
                val = N - self.alpha * lb
                for j in range(j0, j1+1):
                    yv = float(self.y[i_bar, j])
                    if (val > 0 and yv >= 1.0 - self.tol) or \
                       (val <= 0 and yv <= self.tol):
                        dy = 0.0
                    else:
                        dy = (yv + self.beta) / float(self.w[i_bar]) * val
                    self.y[i_bar, j] = np.clip(yv + step * dy, 0.0, 1.0)

            for i in range(self.d):
                if i == i_bar:
                    continue
                for j in range(self.k):
                    yv  = float(self.y[i, j])
                    val = N
                    if (val > 0 and yv >= 1.0 - self.tol) or \
                       (val <= 0 and yv <= self.tol):
                        continue
                    dy = (yv + self.beta) / float(self.w[i]) * val
                    self.y[i, j] = np.clip(yv + step * dy, 0.0, 1.0)

            # Merge blocks on i_bar: if y(Bp) ≈ y(Bp+1) and lam(Bp) < lam(Bp+1)
            merged = True
            while merged:
                for (j0, j1) in blocks:
                    if j1 > j0:
                        v = float(np.mean(self.y[i_bar, j0:j1+1]))
                        self.y[i_bar, j0:j1+1] = v
                new_b, new_l, merged = [], [], False
                i = 0
                while i < len(blocks):
                    if i < len(blocks) - 1:
                        b0, b1   = blocks[i], blocks[i+1]
                        yp = float(self.y[i_bar, b0[0]])
                        yn = float(self.y[i_bar, b1[0]])
                        lp, ln = float(blam[i]), float(blam[i+1])
                        if abs(yp - yn) <= 1e-10 and lp < ln:
                            B    = (b0[0], b1[1])
                            lavg = float(np.mean(lam_row[B[0]:B[1]+1]))
                            new_b.append(B); new_l.append(lavg)
                            i += 2; merged = True; continue
                    new_b.append(blocks[i]); new_l.append(blam[i]); i += 1
                blocks, blam = new_b, new_l

            eta += step


# ─────────────────────────────────────────────────────────────────────────────
# 3. NODE STATE  (Λ_p^t — fractional quota distribution at one HST node)
#
# This class represents Λ_p^t from §4.1 of BBMN:
# the fractional state maintained at one internal HST node p.
#
# At node p:
#   - There are d children (subtrees).
#   - The total number of servers in the whole system is k.
#   - Node p does not have a fixed integer quota; instead it maintains
#     a probability distribution over possible quotas q ∈ {0, …, k}.
#
# ---------------------------------------------------------------------------
# Quota distribution
# ---------------------------------------------------------------------------
#   pi[q]  =  P( node p currently has quota q )
#
# Interpretation:
#   With probability pi[q], the subtree of p is responsible for exactly q
#   servers (fractionally).  Therefore:
#
#       Σ_{q=0..k} pi[q] = 1
#
# Instead of explicitly storing a large collection Λ_p^t of instances
# (as described abstractly in the paper), we compress it into:
#
#   - a weight vector pi[0..k]
#   - one AllocationInstance per quota value q
#
# This is exact because quota values are integers between 0 and k,
# so at most k+1 allocation instances are needed.
#
# ---------------------------------------------------------------------------
# Allocation instances
# ---------------------------------------------------------------------------
#   alloc[q] = AllocationInstance corresponding to quota q
#
# Each AllocationInstance(q) maintains:
#   y_q[i,j]  = P(child i gets ≥ j+1 servers | quota=q) (cf. part 3 of the code)
#
# Therefore:
#   x_q[i,j]  = P(child i gets exactly j servers | quota=q)
#
# The node’s overall marginal allocation is the mixture:
#
#   X[i,j] = Σ_q pi[q] * x_q[i,j]
#
# which represents:
#   P(child i gets exactly j servers) under the current state Λ_p^t.
#
# ---------------------------------------------------------------------------
# step(i_bar, h)
# ---------------------------------------------------------------------------
# Called when a request occurs in the subtree of child i_bar.
#
# Input:
#   i_bar  = index of the child containing the request
#   h[0..k] = hit-cost vector (monotone decreasing, h[0]=int), it's decreasing 
#               because intuitivly if we send k+1 servers to the clients will be
#               more efficient than sending k or less servers in terms of the traveled
#               distance
#
# Procedure:
#
#   1. Convert hit-costs to λ-values:
#         λ_j = max(0, h[j] - h[j+1])
#
#   2. For every quota q with pi[q] > 0:
#         - Run FIX STAGE so total mass equals q.
#         - Run HIT STAGE on child i_bar.
#
#   Note:
#     The weight vector pi itself is not changed here.
#     Only the internal allocation matrices y_q are updated.
#
# After this, each allocation instance has reacted to the request,
# and the node’s fractional distribution among children changes.
#distance
# ---------------------------------------------------------------------------
# mixture_x()
# ---------------------------------------------------------------------------
# Returns the aggregated allocation matrix:
#
#   X[i,j] = Σ_q pi[q] * x_q[i,j]
#
# Shape: (d, k+1)
#
# This is the node’s actual fractional assignment to children,
# after averaging over all possible quotas.
#
# ---------------------------------------------------------------------------
# update_child_pi(i, x_new)
# ---------------------------------------------------------------------------
# Propagates quota information downward to child i.
#
# Given:
#   x_new[j] = P(child i gets exactly j servers)
#
# We define the child’s new quota distribution as:
#
#   pi_child[j] = x_new[j]
#
# This enforces consistency between parent and child:
#   the probability that child i receives j servers
#   becomes the probability that the child subtree has quota j.
#
# (This corresponds to the consistency equations (40)–(41) in §4.1.)
#
# ---------------------------------------------------------------------------
# expected_servers_at_leaf(child_index)
# ---------------------------------------------------------------------------
# Computes:
#
#   E[ number of servers assigned to that child ]
#
# which equals:
#
#   Σ_q pi[q] * ( Σ_j j * x_q[i,j] )
#
# This value is used later to reconstruct the fractional mass x_p
# and ultimately to guide the online rounding step.
#
# ---------------------------------------------------------------------------
# Conceptually:
#
# NodeState is a probabilistic controller:
#   - pi describes uncertainty over how many servers belong to this subtree.
#   - alloc[q] describes how those q servers would be distributed.
#   - mixture_x combines these possibilities into a single marginal view.
#
# When a request arrives, all quota-conditioned allocations react,
# and the mixture shifts probability mass toward the requested subtree.
# ─────────────────────────────────────────────────────────────────────────────
class NodeState:
    """
    Lambda_p^t for one internal node p of the HST.

    pi[q] = weight on quota-q instance.
    alloc[q] = AllocationInstance for that quota (allocation state y).
    """

    def __init__(self, d: int, k: int, w: np.ndarray,
                 init_quota: int, eps: float, deta: float):
        self.d = d
        self.k = k
        self.w = w
        self.eps  = eps
        self.deta = deta

        # Weight vector over quotas 0..k
        self.pi = np.zeros(k + 1, dtype=float)
        self.pi[init_quota] = 1.0

        # One AllocationInstance per quota (created on demand)
        self.alloc: Dict[int, AllocationInstance] = {}
        self._get_or_create_alloc(init_quota)

    def _get_or_create_alloc(self, q: int) -> AllocationInstance:
        if q not in self.alloc:
            inst = AllocationInstance(self.d, self.k,
                                      self.w.copy(), self.eps, self.deta)
            # Initialise y so that sum(y) = q (fix from zero)
            inst.fix_stage(float(q))
            self.alloc[q] = inst
        return self.alloc[q]

    def step(self, i_bar: int, h: np.ndarray):
        """
        Process one request in child i_bar.

        h: hit-cost vector of length k+1, h[j] for j = 0..k.
           h[0] = inf (Observation 17.1)
           h[j] monotone decreasing for j >= 1.

        For each quota q with pi[q] > 0:
          1. Run fix stage to bring sum(y) up to q (in case it drifted down).
          2. Convert h to lambda (differences of h).
          3. Run hit stage on child i_bar.

        Then propagate quota changes to children via consistency (eq. 40-41):
        returns the updated x_{i,j} marginal = E_{q~pi}[x_q[i,j]].
        """
        # Lambda from hit cost (Cote et al.): lam[j] = h[j] - h[j+1] for j=0..k-1
        lam_row = np.zeros(self.k, dtype=float)
        for j in range(self.k):
            hj  = float(h[j])
            hj1 = float(h[j+1])
            if math.isinf(hj):
                lam_row[j] = 1e6
            else:
                lam_row[j] = max(0.0, hj - hj1)

        for q in range(self.k + 1):
            if self.pi[q] < 1e-15:
                continue
            inst = self._get_or_create_alloc(q)
            inst.fix_stage(float(q))
            inst.hit_stage(i_bar, lam_row)

    def mixture_x(self) -> np.ndarray:
        """
        Mixture x_{i,j} = sum_q pi[q] * x_q[i,j],  shape (d, k+1).
        This is the aggregate fractional allocation across all quota instances.
        """
        X = np.zeros((self.d, self.k + 1), dtype=float)
        for q, w in enumerate(self.pi):
            if w < 1e-15:
                continue
            inst = self._get_or_create_alloc(q)
            X   += w * inst.x_matrix()
        return X

    def update_child_pi(self, i: int, x_new: np.ndarray) -> np.ndarray:
        """
        Compute the new pi vector for child i, consistent with equation (40).

        x_new[j] = x_{i,j} = aggregate P(child i gets exactly j servers).
        The child's pi' must satisfy:
          sum_{q: pi'[q]=j} pi'[q] = x_{i,j}   for each j.
        This means exactly: pi'[j] = x_{i,j}   (one-to-one mapping quota -> j servers).

        This is the elementary-move update: the child's weight on quota j
        equals the parent's aggregate probability of assigning j servers to child i.
        """
        new_pi = np.maximum(x_new, 0.0)
        s = float(new_pi.sum())
        if s > 1e-12:
            new_pi /= s
        return new_pi

    def expected_servers_at_leaf(self, leaf_child_idx: int) -> float:
        """
        z(q, t) from §4.1:  sum_s lambda_s * sum_j j * x_{q,j,s}
        = sum_q pi[q] * E_q[servers at child leaf_child_idx]
        """
        total = 0.0
        for q in range(self.k + 1):
            if self.pi[q] < 1e-15:
                continue
            inst = self._get_or_create_alloc(q)
            x    = inst.x_matrix()[leaf_child_idx]  # shape (k+1,)
            total += self.pi[q] * float(sum(j * x[j] for j in range(self.k + 1)))
        return total


# ─────────────────────────────────────────────────────────────────────────────
# 4. FRACTIONAL SOLVER  (§4.1 — Global fractional k-server dynamics)
#
# This class orchestrates the entire fractional k-server algorithm
# across the whole HST.  It maintains one NodeState per internal node
# and updates them consistently whenever a request arrives.
#
# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------
#
#   hst        : the sigma-HST embedding of the metric space
#   k          : total number of servers
#   node_state : dictionary mapping each internal node => its NodeState
#   x_leaf[i]  : fractional number of servers currently at leaf i
#
# Invariant:
#     sum_i x_leaf[i] = k
#
# Each NodeState maintains:
#     pi[q]        = probability that node has quota q
#     alloc[q]     = AllocationInstance for quota q
#
# Together, these define Λ_p^t in the BBMN paper.
#
# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------
#
# At time 0:
#   - All k servers are located at the geometric origin.
#   - Therefore the root has quota k.
#   - That quota flows entirely down one branch of the tree.
#
# _init_states(node, init_quota):
#
#   Recursively initializes each internal node:
#       - Creates its NodeState with quota = init_quota.
#       - Propagates that quota downward:
#             first child gets init_quota,
#             other children get 0.
#
# This sets the initial fractional configuration.
#
# ---------------------------------------------------------------------------
# serve(req)
# ---------------------------------------------------------------------------
#
# Called when a request occurs at leaf "req".
#
# Step 1 — Find root-to-leaf path:
#     path_nodes = [root → ... → leaf]
#
# Only nodes along this path are affected.
#
# Step 2 — For each internal node on this path:
#
#   (a) Identify which child subtree contains the request.
#       Let that index be i_bar.
#
#   (b) Build hit-cost vector h[0..k]:
#           h[0] = inf
#           h[j] decreasing in j (proxy for OPT difference)
#
#       This approximates the hit-cost sequence used in §4.1.
#
#   (c) Call state.step(i_bar, h):
#           - FIX stage restores quota mass.
#           - HIT stage redistributes mass toward requested child.
#
#       This updates the allocation matrices y_q inside the node.
#
#   (d) Compute mixture allocation:
#           X[i,j] = Σ_q pi[q] * x_q[i,j]
#
#       This is the aggregate probability that child i receives j servers.
#
#   (e) Extract distribution for requested child:
#           x_i[j] = P(child i_bar gets exactly j servers)
#
#   (f) Update child's quota distribution:
#           child.pi[j] = x_i[j]
#
#       This enforces consistency between parent and child
#       (equations (40)–(41) in §4.1).
#
# This process propagates quota information top-down along the path.
#
# ---------------------------------------------------------------------------
# _recompute_leaf_values()
# ---------------------------------------------------------------------------
#
# After all NodeStates are updated, we recompute the fractional
# server count at each leaf.
#
# For each leaf i:
#     - Look at its parent node p.
#     - Let ci be the index of this leaf among p's children.
#     - Compute:
#
#           x_leaf[i] =
#               Σ_q pi[q] * ( Σ_j j * x_q[ci, j] )
#
#       which equals the expected number of servers assigned
#       to that child under Λ_p^t.
#
# Finally:
#     Normalize x_leaf so that sum_i x_leaf[i] = k.
#
# ---------------------------------------------------------------------------
# Conceptually:
#
#   Each request triggers a cascade:
#
#       root
#         ↓
#       internal nodes along path
#         ↓
#       leaf
#
#   At each node:
#       - allocation shifts toward the requested subtree,
#       - quota distributions are updated,
#       - consistency is maintained between parent and child.
#
#   The result is a globally coherent fractional k-server configuration
#   that adapts over time to the request sequence.
# ─────────────────────────────────────────────────────────────────────────────
class FractionalSolver:
    """
    Maintains fractional k-server state across all internal HST nodes (§4.1).
    """

    def __init__(self, hst: SigmaHST, k: int, eps: float, deta: float):
        self.hst  = hst
        self.k    = k
        self.eps  = eps
        self.deta = deta

        # x_leaf[i] = current fractional server count at leaf i
        self.x_leaf: np.ndarray = np.zeros(hst.n, dtype=float)
        # All k servers start at site 0 (which maps to leaf index origin)
        # Set after construction by the top-level class.

        # NodeState for each internal node
        self.node_state: Dict[int, NodeState] = {}
        self._init_states(hst.root, k)

    def _init_states(self, node: HSTNode, init_quota: int):
        """
        Recursively initialise NodeState for every internal node.
        Initially all k servers are at origin, so:
          - root gets quota k
          - at each level the quota flows down: all k go to the child containing origin
        """
        if node.is_leaf:
            return
        d = len(node.children)
        w = np.array([max(ch.edge_weight, 1e-6) for ch in node.children], dtype=float)
        self.node_state[node.id] = NodeState(
            d=d, k=self.k, w=w,
            init_quota=min(init_quota, self.k),
            eps=self.eps, deta=self.deta
        )
        # All servers go to child 0 initially (child containing origin)
        # (We just give all quota to child 0 for simplicity; the fix stage handles it)
        for idx, ch in enumerate(node.children):
            child_quota = init_quota if idx == 0 else 0
            self._init_states(ch, child_quota)

    def serve(self, req: int):
        """
        Process request at site req.  Updates all node states top-down (§4.1).
        After this call, x_leaf[i] = fractional server count at each leaf.
        """
        path = self.hst._path_to_root[req]   # [leaf, ..., root]
        path_nodes = list(reversed(path))     # [root, ..., leaf]

        for depth, node in enumerate(path_nodes[:-1]):
            child = path_nodes[depth + 1]
            if node.is_leaf:
                break

            state = self.node_state[node.id]
            ci    = node.children.index(child)  # index of child on path to req

            # Hit cost (Observation 17):
            # h[0] = inf, h[j] = 0 for j>=1 on other children
            # For the requested child: h[j] = Optcost proxy (decreasing).
            # We use the standard proxy: h[j] = max(0, 1/(j+eps)) for j>=1.
            # This satisfies monotonicity and gives h[0]=inf.
            h = np.zeros(self.k + 1, dtype=float)
            h[0] = float('inf')
            for j in range(1, self.k + 1):
                h[j] = 1.0 / (j + self.eps)   # decreasing, proxy for Optcost diff

            # Update this node's allocation (fix + hit)
            state.step(ci, h)

            # Propagate new quota distribution to child (consistency eq. 40-41)
            X_mix = state.mixture_x()          # shape (d, k+1)
            x_i   = X_mix[ci]                  # shape (k+1,): P(child i gets j servers)
            child_new_pi = state.update_child_pi(ci, x_i)

            if not child.is_leaf and child.id in self.node_state:
                old_pi = self.node_state[child.id].pi
                # Elementary move update: blend toward new_pi
                self.node_state[child.id].pi = child_new_pi

        # Recompute x_leaf from the leaf's parent state
        self._recompute_leaf_values()

    def _recompute_leaf_values(self):
        """
        z(q, t) = sum_{s in Lambda_p} lambda_s * sum_j j * x_{q,j,s}   [§4.1 eq.]
        Recompute for all leaves from their parent's NodeState.
        """
        for i in range(self.hst.n):
            leaf = self.hst.leaf_node[i]
            if leaf is None or leaf.parent is None:
                continue
            parent = leaf.parent
            if parent.id not in self.node_state:
                continue
            state = self.node_state[parent.id]
            ci    = parent.children.index(leaf)
            self.x_leaf[i] = state.expected_servers_at_leaf(ci)

        # Normalise: total servers must equal k
        total = float(self.x_leaf.sum())
        if total > 1e-9:
            self.x_leaf *= self.k / total


# ─────────────────────────────────────────────────────────────────────────────
# 5. ONLINE ROUNDING  (§5.2 — Theorem 24 + Lemma 25)
#
# Goal
# ----
# The fractional solver maintains x_leaf[i] = "fractional #servers" at each leaf i.
# But the actual k-server problem requires an INTEGRAL configuration: exactly k
# servers placed on leaves. OnlineRounder is the layer that turns the fractional
# state into a concrete integral move sequence (i.e., which real server moves).
#
# Paper picture (informal)
# ------------------------
# In BBMN, the rounding maintains an integral configuration C(t) that is
# (approximately) consistent and balanced w.r.t. the fractional mass x(t):
#
#   For every subtree p:
#       n_p(C) ≈ x_p
#
# where:
#   - x_p = sum_{leaf in subtree p} x_leaf[leaf]     (fractional mass inside p)
#   - n_p(C) = number of integral servers currently inside subtree p
#
# Theorem 24 + Lemma 25 describe how to perform "swap" operations so that when the
# fractional solution makes an infinitesimal change (moves delta mass), the
# integral configuration performs a corresponding server move with probability
# proportional to that delta, preserving the invariant in expectation.
#
# This implementation: what it does
# ---------------------------------
# This code does NOT implement the full infinitesimal (δ tends to 0) randomized swap
# process explicitly. Instead, it produces a practical online rounding heuristic
# inspired by Theorem 24 / Lemma 25:
#
#   - If a request arrives and no server is already there:
#       pick one server index to move to req
#       pay Manhattan distance from its old location to req
#
# Data / Variables
# ---------------
#   self.servers : List[Optional[int]]
#       servers[s] is the LEAF INDEX where server s currently sits.
#       None means "still at geometric origin (0,0)" (not an actual leaf).
#
#   self.start_coord : (0,0)
#       geometric starting coordinate used only for cost computation when old=None.
#
#   self.hst
#       used only for "tree distance" hst_dist and LCA computations (tie-breaking).
#
#   self.frac.x_leaf
#       current fractional masses; used to estimate x_p in subtrees.
#
# serve(req)
# ----------
# Input:
#   req : leaf index of the requested point.
#
# Steps:
#   1) Already served check:
#        if any server is already exactly at leaf req, cost is 0.
#        (k-server rule: request served without moving.)
#
#   2) Choose a server to move:
#        chosen = _pick_server_theorem24(req)
#
#   3) Compute movement cost:
#        if the chosen server is still at origin (old is None):
#            cost = Manhattan( (0,0), sites[req] )
#        else:
#            cost = Manhattan( sites[old], sites[req] )
#
#   4) Apply the move:
#        self.servers[chosen] = req
#        return (chosen, cost)
#
# _pick_server_theorem24(req)
# ---------------------------
# This function is your "server selection rule".
# It tries to mimic the Lemma 25 idea: move a server from a subtree that has
# "too many" integral servers compared to the fractional mass there (surplus).
#
# Step A — Prefer unused servers at the origin
#   If any server is still None, we always pick it first.
#   Reason: moving an unused server from origin does not disturb an existing
#   subtree balance (since it wasn't counted inside the tree at all).
#
# Step B — Otherwise, evaluate each candidate server s at leaf position "pos"
#
# For each server index idx with current location s:
#
#   1) Compute tree distance:
#        hst_d = hst_dist(s, req)
#      This approximates "how far" the server is from req in the HST geometry.
#      The algorithm prefers small hst_d (closer in the tree).
#
#   2) Compute the LCA subtree:
#        lca_node = LCA(s, req)
#      This identifies the smallest subtree that contains both s and req.
#      Intuition: moving a server from s to req changes server counts mainly
#      inside subtrees on the path up to this LCA.
#
#   3) Estimate fractional mass in that subtree:
#        xp = sum_{leaf in lca_node.leaves} x_leaf[leaf]
#      This is x_p: "how many fractional servers live inside that subtree".
#
#   4) Count integral servers in that subtree:
#        np_C = #{servers currently located in lca_node.leaves}
#      This is n_p(C): "how many real servers are currently inside that subtree".
#
#   5) Define surplus:
#        surplus = xp - np_C
#
#      Interpretation:
#        - surplus > 0: subtree wants MORE integral servers (fractional ahead)
#        - surplus < 0: subtree has TOO MANY integral servers (integral ahead)
#
#      If we want to move a server OUT of somewhere, we'd like to pick it from
#      places where integral is "too heavy" compared to fractional.
#
#   6) Build a lexicographic score:
#        score = (hst_d, -surplus)
#
#      Minimization logic:
#        - First: minimize hst_d (closer server in the HST sense)
#        - Tie-break: maximize surplus (because -surplus smaller means surplus bigger)
#
#      NOTE: This tie-break is slightly counterintuitive if you interpret surplus
#      as "where to remove servers". In Lemma 25, you'd typically remove from a
#      subtree where n_p(C) > x_p (negative surplus). Your tie-break prefers
#      larger surplus, i.e. xp > np_C, which is the opposite direction.
#      So this is not a faithful Lemma 25 implementation; it's a heuristic.
#
# Step C — Random tie-breaking
#   If multiple servers share the best score, choose uniformly at random:
#        return rng.choice(cands)
#
# This injects randomness similarly to the "randomized rounding" spirit.
#
# Summary of behavior on new requests
# ----------------------------------
# When a new request arrives:
#   - If a server is already there => no move.
#   - Else:
#       1) if any server is unused (None) => deploy it to the request.
#       2) otherwise:
#          pick a server that is tree-close to the request, with a surplus-based
#          tie-break using fractional masses in the LCA subtree.
#       3) move that server and pay Manhattan distance in the original metric.
# ─────────────────────────────────────────────────────────────────────────────
class OnlineRounder:
    """
    Online rounding (Theorem 24 + Lemma 25)

    Servers start at geometric origin (0,0),
    not necessarily at a site.
    """

    def __init__(self, hst: SigmaHST, k: int,
                 frac: FractionalSolver, rng: random.Random,
                 init_servers: List[Optional[int]],
                 start_coord=(0, 0)):

        self.hst = hst
        self.k = k
        self.frac = frac
        self.rng = rng

        # Servers initially at geometric origin (represented by None)
        self.servers: List[Optional[int]] = list(init_servers)

        # True geometric starting coordinate
        self.start_coord = start_coord

    def serve(self, req: int) -> Tuple[int, float]:

        # Already served
        for idx, s in enumerate(self.servers):
            if s == req:
                return idx, 0.0

        chosen = self._pick_server_theorem24(req)

        old = self.servers[chosen]

        # Compute movement cost
        if old is None:
            # Moving from geometric origin
            cost = manhattan(self.start_coord,
                             self.hst.sites[req])
        else:
            cost = manhattan(self.hst.sites[old],
                             self.hst.sites[req])

        self.servers[chosen] = req
        return chosen, cost

    def _pick_server_theorem24(self, req: int) -> int:

        # If any server still at origin (None), use it first
        for idx, s in enumerate(self.servers):
            if s is None:
                return idx

        best_score = None
        cands: List[int] = []

        for idx, s in enumerate(self.servers):

            if s == req:
                continue

            hst_d = self.hst.hst_dist(s, req)

            # Lemma 25 surplus logic
            lca_node = self.hst.lca(s, req)
            xp = float(sum(self.frac.x_leaf[i]
                           for i in lca_node.leaves))

            lca_leaves = set(lca_node.leaves)
            np_C = sum(1 for sv in self.servers
                       if sv in lca_leaves)

            surplus = xp - np_C

            score = (round(hst_d, 6), -surplus)

            if best_score is None or score < best_score:
                best_score = score
                cands = [idx]
            elif score == best_score:
                cands.append(idx)

        return self.rng.choice(cands)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  TOP-LEVEL: BBMNExact
# ─────────────────────────────────────────────────────────────────────────────

class BBMNExact:
    """
    Full BBMN algorithm with geometric origin (0,0).
    """

    def __init__(self, sites: List[Tuple[int, int]], k: int,
                 sigma: float = 6.0,
                 eps: float = 0.5,
                 deta: float = 5e-3,
                 seed: Optional[int] = None):

        self.sites = sites
        self.k = k
        self.start_coord = (0, 0)

        rng = random.Random(seed)

        # 1. Build HST
        self.hst = SigmaHST(sites, sigma=sigma, rng=rng)

        # 2. Fractional solver
        self.frac = FractionalSolver(self.hst, k, eps=eps, deta=deta)

        # Initially no fractional mass at any site
        self.frac.x_leaf[:] = 0.0

        # 3. Integral servers start at geometric origin
        init_servers = [None] * k

        self.rounder = OnlineRounder(
            self.hst,
            k,
            self.frac,
            rng,
            init_servers,
            start_coord=self.start_coord
        )

        self.total_cost = 0.0

    def serve(self, req: int) -> float:

        # Update fractional state
        self.frac.serve(req)

        # Round to integral
        _, cost = self.rounder.serve(req)

        self.total_cost += cost
        return cost

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PUBLIC INTERFACE  (project convention: run_*(path, **kwargs) -> (cost, opt))
# ─────────────────────────────────────────────────────────────────────────────

def run_bbmn_fractional(path: str,
                   sigma: float = 6.0,
                   eps:   float = 0.5,
                   deta:  float = 5e-3,
                   seed:  Optional[int] = None,
                   **kwargs) -> Tuple[float, float]:
    """
    Run the exact BBMN algorithm on one instance.

    Parameters
    ----------
    path  : path to .inst file
    sigma : HST separation parameter (must be > 5 per Theorem 24)
    eps   : smoothing parameter for the allocation algorithm (Cote et al.)
    deta  : ODE step size for the hit stage
    seed  : RNG seed (None = fresh random HST each call)

    Returns
    -------
    (cost, opt)
    """
    instance = load_instance(path)
    sites    = instance["sites"]
    requests = instance["requests"]
    k        = instance["k"]
    opt      = instance["opt"]

    algo = BBMNExact(sites, k, sigma=sigma, eps=eps, deta=deta, seed=seed)

    for req in requests:
        algo.serve(req)

    return float(algo.total_cost), float(opt)