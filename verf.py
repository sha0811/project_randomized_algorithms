import math
from algorithms.polylogarithmic_randomized import BBMNExact
from utilities import load_instance, manhattan


# ─────────────────────────────────────────────────────────────
# Exact Offline OPT (small instances only)
# ─────────────────────────────────────────────────────────────

def offline_optimal_exact(sites, requests, k, max_states=2_000_000):
    """
    Exact offline optimal k-server cost via DP.
    Servers start at geometric origin (0,0).
    """

    start_state = tuple([None] * k)
    dp = {start_state: 0.0}

    origin = (0, 0)

    for t, req in enumerate(requests):
        new_dp = {}

        for state, cost_so_far in dp.items():

            # Already served
            if req in state:
                best = new_dp.get(state, float("inf"))
                if cost_so_far < best:
                    new_dp[state] = cost_so_far
                continue

            # Try moving each server
            for i in range(k):
                old = state[i]

                if old is None:
                    move_cost = manhattan(origin, sites[req])
                else:
                    move_cost = manhattan(sites[old], sites[req])

                next_state = list(state)
                next_state[i] = req
                next_state = tuple(sorted(next_state, key=lambda x: -1 if x is None else x))

                cand = cost_so_far + move_cost
                prev = new_dp.get(next_state, float("inf"))
                if cand < prev:
                    new_dp[next_state] = cand

        dp = new_dp

        if len(dp) > max_states:
            raise RuntimeError("Exact OPT infeasible (state explosion).")

    return min(dp.values())


# ─────────────────────────────────────────────────────────────
# FULL TRACE BBMN
# ─────────────────────────────────────────────────────────────

def trace_bbmn(path, sigma=6.0, eps=0.05, deta=0.05, seed=0):

    inst = load_instance(path)
    sites = inst["sites"]
    requests = inst["requests"]
    k = inst["k"]
    opt = inst.get("opt", None)

    algo = BBMNExact(sites, k, sigma=sigma, eps=eps, deta=deta, seed=seed)

    total_cost = 0.0
    origin = (0, 0)

    print("\n" + "=" * 70)
    print(f"Instance: {path}")
    print(f"k = {k}")
    print(f"#requests = {len(requests)}")
    print(f"OPT (from file) = {opt}")
    print("=" * 70)

    for t, req in enumerate(requests):

        print(f"\nRequest {t}:")
        print(f"  Requested site index: {req}")
        print(f"  Coordinates: {sites[req]}")
        print(f"  Current servers: {algo.rounder.servers}")

        # Already served
        if req in algo.rounder.servers:
            print("  -> Already served (cost = 0)")
            continue

        # Update fractional state
        algo.frac.serve(req)

        # Choose server
        chosen = algo.rounder._pick_server_theorem24(req)
        old = algo.rounder.servers[chosen]

        # Compute cost
        if old is None:
            cost = manhattan(origin, sites[req])
            print(f"  -> Moving server {chosen}")
            print(f"     from origin {origin}")
        else:
            cost = manhattan(sites[old], sites[req])
            print(f"  -> Moving server {chosen}")
            print(f"     from site {old} {sites[old]}")

        print(f"     to   site {req} {sites[req]}")
        print(f"     Manhattan cost = {cost}")

        # Apply move
        algo.rounder.servers[chosen] = req
        total_cost += cost

        print(f"  Updated servers: {algo.rounder.servers}")
        print(f"  Running total cost: {total_cost}")

    print("\n" + "=" * 70)
    print(f"FINAL TOTAL COST (BBMN) = {total_cost}")
    print("=" * 70)

    if opt is not None and opt > 0:
        print(f"FILE OPT = {opt}")
        print(f"FILE RATIO = {total_cost / opt:.4f}")

    # Try exact OPT if feasible
    try:
        n = len(sites)
        T = len(requests)

        if (n ** k) * T <= 200_000 and k <= 4 and T <= 60:
            print("\nComputing EXACT OFFLINE OPT...")
            opt_exact = offline_optimal_exact(sites, requests, k)
            print(f"EXACT OFFLINE OPT = {opt_exact}")
            print(f"TRUE RATIO = {total_cost / opt_exact:.4f}")
        else:
            print("\nExact offline OPT skipped (instance too large).")

    except Exception as e:
        print("Exact offline OPT failed:", e)

    print("=" * 70)

    return total_cost


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    trace_bbmn(sys.argv[1])