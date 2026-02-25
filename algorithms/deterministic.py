import math
from utilities import load_instance, manhattan, move_toward


def run_greedy(path, **kwargs):
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        # finding position of site corresponding to current request
        site_pos = clients[request]
        # choosing server to assign to current request
        server = choose_server_greedy(servers, site_pos, state, **kwargs)
        # updating cost
        cost += manhattan(servers[server], site_pos)
        # updating position of server
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_greedy(servers, site_pos, state, **kwargs):
    best_server = 0
    best_dist = manhattan(servers[0], site_pos)
    for j in range(1, len(servers)):
        dist = manhattan(servers[j], site_pos)
        if dist < best_dist:
            best_dist = dist
            best_server = j
    return best_server


def run_balance(path, alpha=0.5, **kwargs):
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        # finding position of site corresponding to current request
        site_pos = clients[request]
        # choosing server to assign to current request
        server = choose_server_balance(servers, site_pos, state, alpha=alpha, **kwargs)
        # updating cost
        cost += manhattan(servers[server], site_pos)
        # updating position of server
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_balance(servers, site_pos, state, alpha=0.5, **kwargs):
    """
    Chooses the server that minimizes: distance(server, request) + alpha * (distance already traveled by this server).
    total_dist: list of length k, total_dist[j] = total distance already traveled by server j.
    alpha: weight of the charge.
    """
    if "total_dist" not in state:
        state["total_dist"] = [0] * len(servers)
    best_server = 0
    best_score = manhattan(servers[0], site_pos) + alpha * state["total_dist"][0]
    for j in range(1, len(servers)):
        score = manhattan(servers[j], site_pos) + alpha * state["total_dist"][j]
        if score < best_score:
            best_score = score
            best_server = j
    d = manhattan(servers[best_server], site_pos)
    state["total_dist"][best_server] += d
    return best_server


def run_balance_aggressive(path, alpha=0.01, **kwargs):
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        # finding position of site corresponding to current request
        site_pos = clients[request]
        # choosing server to assign to current request
        server = choose_server_balance_aggressive(servers, site_pos, state, alpha=alpha, **kwargs)
        # updating cost
        cost += manhattan(servers[server], site_pos)
        # updating position of server
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_balance_aggressive(servers, site_pos, state, alpha=0.01, **kwargs):
    """
    Like Balance but with quadratic penalty on total_dist: in the hope of encouraging even more spreading.
    score = distance + alpha * (total_dist[j])^2
    """
    if "total_dist" not in state:
        state["total_dist"] = [0] * len(servers)
    best_server = 0
    best_score = manhattan(servers[0], site_pos) + alpha * (state["total_dist"][0] ** 2)
    for j in range(1, len(servers)):
        score = manhattan(servers[j], site_pos) + alpha * (state["total_dist"][j] ** 2)
        if score < best_score:
            best_score = score
            best_server = j
    d = manhattan(servers[best_server], site_pos)
    state["total_dist"][best_server] += d
    return best_server


def run_balance_exponential(path, alpha=0.5, beta=0.01, **kwargs):
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        # finding position of site corresponding to current request
        site_pos = clients[request]
        # choosing server to assign to current request
        server = choose_server_balance_exponential(servers, site_pos, state, alpha=alpha, beta=beta, **kwargs)
        # updating cost
        cost += manhattan(servers[server], site_pos)
        # updating position of server
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_balance_exponential(servers, site_pos, state, alpha=0.5, beta=0.01, **kwargs):
    """
    Balance but this time we use an exponential penalty on total_dist.
    score = distance + alpha * exp(beta * total_dist[j])
    """
    if "total_dist" not in state:
        state["total_dist"] = [0] * len(servers)
    best_server = 0
    best_score = manhattan(servers[0], site_pos) + alpha * math.exp(beta * state["total_dist"][0])
    for j in range(1, len(servers)):
        score = manhattan(servers[j], site_pos) + alpha * math.exp(beta * state["total_dist"][j])
        if score < best_score:
            best_score = score
            best_server = j
    d = manhattan(servers[best_server], site_pos)
    state["total_dist"][best_server] += d
    return best_server


def run_balance_sqrt(path, alpha=0.5, **kwargs):
    """Other try: Balance with sublinear (sqrt) penalty on total distance"""
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        site_pos = clients[request]
        server = choose_server_balance_sqrt(servers, site_pos, state, alpha=alpha, **kwargs)
        cost += manhattan(servers[server], site_pos)
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_balance_sqrt(servers, site_pos, state, alpha=0.5, **kwargs):
    """Like Balance but penalty is alpha * sqrt(total_dist[j])."""
    if "total_dist" not in state:
        state["total_dist"] = [0] * len(servers)
    best_server = 0
    best_score = manhattan(servers[0], site_pos) + alpha * math.sqrt(state["total_dist"][0])
    for j in range(1, len(servers)):
        score = manhattan(servers[j], site_pos) + alpha * math.sqrt(state["total_dist"][j])
        if score < best_score:
            best_score = score
            best_server = j
    d = manhattan(servers[best_server], site_pos)
    state["total_dist"][best_server] += d
    return best_server


def run_balance_time_decay(path, alpha_0=0.8, decay=0.99995, **kwargs):
    """Decreasing penalty on already travelled distance over time: alpha_t = alpha_0 * decay^t."""
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        site_pos = clients[request]
        server = choose_server_balance_time_decay(
            servers, site_pos, state, alpha_0=alpha_0, decay=decay, **kwargs
        )
        cost += manhattan(servers[server], site_pos)
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_balance_time_decay(servers, site_pos, state, alpha_0=0.8, decay=0.99995, **kwargs):
    """
    Balance with alpha that decays over time: alpha_t = alpha_0 * decay^t.
    Early requests favour load balancing and later requests favour nearest server.
    """
    if "total_dist" not in state:
        state["total_dist"] = [0] * len(servers)
    if "t" not in state:
        state["t"] = 0
    t = state["t"]
    state["t"] += 1
    alpha = alpha_0 * (decay ** t)
    best_server = 0
    best_score = manhattan(servers[0], site_pos) + alpha * state["total_dist"][0]
    for j in range(1, len(servers)):
        score = manhattan(servers[j], site_pos) + alpha * state["total_dist"][j]
        if score < best_score:
            best_score = score
            best_server = j
    d = manhattan(servers[best_server], site_pos)
    state["total_dist"][best_server] += d
    return best_server


def run_balance_time_decay_inverse(path, alpha_0=0.6, T=8000, **kwargs):
    """
    Balance with inverse time decay: alpha_t = alpha_0 / (1 + t/T).
    It has a smoother decay than exponential.
    """
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        site_pos = clients[request]
        server = choose_server_balance_time_decay_inverse(
            servers, site_pos, state, alpha_0=alpha_0, T=T, **kwargs
        )
        cost += manhattan(servers[server], site_pos)
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_balance_time_decay_inverse(servers, site_pos, state, alpha_0=0.6, T=8000, **kwargs):
    if "total_dist" not in state:
        state["total_dist"] = [0] * len(servers)
    if "t" not in state:
        state["t"] = 0
    t = state["t"]
    state["t"] += 1
    alpha = alpha_0 / (1.0 + t / T)
    best_server = 0
    best_score = manhattan(servers[0], site_pos) + alpha * state["total_dist"][0]
    for j in range(1, len(servers)):
        score = manhattan(servers[j], site_pos) + alpha * state["total_dist"][j]
        if score < best_score:
            best_score = score
            best_server = j
    d = manhattan(servers[best_server], site_pos)
    state["total_dist"][best_server] += d
    return best_server


# ── Balance time decay + reuse only when cost 0  ─────────────


def run_balance_reuse_zero(path, alpha_0=0.55, decay=0.999, **kwargs):
    """
    Balance with time-decaying alpha. If a server is already at the request (cost 0),
    prefers the one that last served this site to stabilize assignment. Otherwise same as balance_time_decay.
    """
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for site_idx in requests:
        site_pos = clients[site_idx]
        server = choose_server_balance_reuse_zero(
            servers, site_pos, site_idx, state, alpha_0=alpha_0, decay=decay, **kwargs
        )
        cost += manhattan(servers[server], site_pos)
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_balance_reuse_zero(servers, site_pos, site_idx, state, alpha_0=0.55, decay=0.999, **kwargs):
    if "total_dist" not in state:
        state["total_dist"] = [0] * len(servers)
    if "last_server_for_site" not in state:
        state["last_server_for_site"] = {}
    if "t" not in state:
        state["t"] = 0
    t = state["t"]
    state["t"] += 1
    k = len(servers)
    dists = [manhattan(servers[j], site_pos) for j in range(k)]
    min_d = min(dists)
    if min_d == 0:
        last = state["last_server_for_site"].get(site_idx)
        if last is not None and dists[last] == 0:
            state["last_server_for_site"][site_idx] = last
            return last
        for j in range(k):
            if dists[j] == 0:
                state["last_server_for_site"][site_idx] = j
                return j
    alpha = alpha_0 * (decay ** t)
    best_server = 0
    best_score = dists[0] + alpha * state["total_dist"][0]
    for j in range(1, k):
        score = dists[j] + alpha * state["total_dist"][j]
        if score < best_score:
            best_score = score
            best_server = j
    state["total_dist"][best_server] += dists[best_server]
    state["last_server_for_site"][site_idx] = best_server
    return best_server


# ── Site affinity: prefer reusing the server that last served this site ─────────


def run_site_affinity(path, alpha=0.5, max_ratio=1.4, **kwargs):
    """
    When the same site is requested again, prefer the server that last served it
    if that server is within max_ratio * (min distance); else use balance.
    """
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for site_idx in requests:
        site_pos = clients[site_idx]
        server = choose_server_site_affinity(
            servers, site_pos, site_idx, state, alpha=alpha, max_ratio=max_ratio, **kwargs
        )
        cost += manhattan(servers[server], site_pos)
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_site_affinity(servers, site_pos, site_idx, state, alpha=0.5, max_ratio=1.4, **kwargs):
    """Reuse the server that last served this site when already at site (cost 0) or within max_ratio * min_dist."""
    if "total_dist" not in state:
        state["total_dist"] = [0] * len(servers)
    if "last_server_for_site" not in state:
        state["last_server_for_site"] = {}
    k = len(servers)
    dists = [manhattan(servers[j], site_pos) for j in range(k)]
    min_dist = min(dists)
    if min_dist == 0:
        last = state["last_server_for_site"].get(site_idx)
        if last is not None and dists[last] == 0:
            state["last_server_for_site"][site_idx] = last
            return last
        for j in range(k):
            if dists[j] == 0:
                state["last_server_for_site"][site_idx] = j
                return j
    last = state["last_server_for_site"].get(site_idx)
    if last is not None and dists[last] <= max_ratio * min_dist:
        state["total_dist"][last] += dists[last]
        state["last_server_for_site"][site_idx] = last
        return last
    best_server = 0
    best_score = dists[0] + alpha * state["total_dist"][0]
    for j in range(1, k):
        score = dists[j] + alpha * state["total_dist"][j]
        if score < best_score:
            best_score = score
            best_server = j
    state["total_dist"][best_server] += dists[best_server]
    state["last_server_for_site"][site_idx] = best_server
    return best_server


def run_double_coverage(path, **kwargs):
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        # finding position of site corresponding to current request
        site_pos = clients[request]
        # choosing server to assign to current request
        result = choose_server_double_coverage(servers, site_pos, state, **kwargs)
        cost_inc, updates = result
        cost += cost_inc
        for j, pos in updates.items():
            servers[j] = pos
    return (cost, instance["opt"])


def choose_server_double_coverage(servers, site_pos, state, **kwargs):
    """
    Double Coverage (2D Manhattan): we move the two nearest servers toward the client until one reaches it.
    Returns (cost, updates_dict) and then the eval loop can update both servers and the cost.
    Cost = 2 * min(d1, d2).
    """
    n = len(servers)
    if n < 2:
        d0 = manhattan(servers[0], site_pos)
        return (d0, {0: site_pos})
    dists = [(manhattan(servers[j], site_pos), j) for j in range(n)]
    dists.sort(key=lambda x: x[0])
    d1, j1 = dists[0]
    d2, j2 = dists[1]
    move_dist = d1
    cost = 2 * move_dist
    updates = {j1: site_pos}
    updates[j2] = move_toward(servers[j2], site_pos, move_dist)
    return (cost, updates)


# ── Work Function Algorithm (WFA) ─────────────────────────────────────────────


def run_wfa(path, **kwargs):
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        site_pos = clients[request]
        move_cost, servers = choose_server_wfa(servers, site_pos, state, **kwargs)
        cost += move_cost
    return (cost, instance["opt"])


def choose_server_wfa(servers, site_pos, state, max_configs=500, **kwargs):
    """
    Work Function Algorithm (WFA) — theoretically (2k-1)-competitive.

    The work function w(t, S) is the minimum cost to serve requests r_1..r_t
    and end up with servers in configuration S (a sorted tuple of k positions).

    At each step t with new request r:
      1. Update the work function: for each known configuration S and each
         server position p in S, we can move p to r at cost d(p, r), yielding
         configuration S' = S with p replaced by r. We keep the minimum cost
         over all ways to reach S'.
      2. Choose server j that minimises:
             w'(current_config with s_j replaced by r) + d(s_j, r)
         where w' is the freshly computed work function. This is the WFA rule.

    To stay tractable over long sequences, we prune the work function after
    each step: we keep at most `max_configs` configurations with the lowest
    work function values, plus always the current server configuration.

    State stores:
      - "wf": dict mapping config (sorted tuple) -> float (work function value)
      - "k": number of servers
    """
    if "wf" not in state:
        init = tuple(sorted(servers))
        state["wf"] = {init: 0.0}
        state["k"] = len(servers)

    k = state["k"]
    wf = state["wf"]

    # ── Step 1: update work function after request site_pos ──────────────────
    new_wf = {}
    for config, w in wf.items():
        config_list = list(config)
        seen_positions = set()
        for i, pos in enumerate(config_list):
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            d = manhattan(pos, site_pos)
            new_config = tuple(sorted(config_list[:i] + [site_pos] + config_list[i + 1:]))
            new_cost = w + d
            if new_config not in new_wf or new_wf[new_config] > new_cost:
                new_wf[new_config] = new_cost

    # ── Step 2: WFA server selection ─────────────────────────────────────────
    servers_list = list(servers)
    current_config = tuple(sorted(servers_list))
    best_j = 0
    best_score = math.inf
    seen_positions = set()
    for j, pos in enumerate(servers_list):
        if pos in seen_positions:
            continue
        seen_positions.add(pos)
        d = manhattan(pos, site_pos)
        idx = list(current_config).index(pos)
        candidate = tuple(sorted(list(current_config[:idx]) + [site_pos] + list(current_config[idx + 1:])))
        w_prime = new_wf.get(candidate, math.inf)
        score = w_prime + d
        if score < best_score:
            best_score = score
            best_j = j

    move_cost = manhattan(servers_list[best_j], site_pos)
    servers_list[best_j] = site_pos

    # ── Step 3: prune work function to stay tractable ─────────────────────────
    # Always keep the current (new) configuration; prune the rest by cost.
    new_current = tuple(sorted(servers_list))
    if len(new_wf) > max_configs:
        keep = sorted(new_wf.items(), key=lambda x: x[1])[:max_configs]
        new_wf = dict(keep)
        # Guarantee the current config is always present
        if new_current not in new_wf:
            new_wf[new_current] = best_score

    state["wf"] = new_wf
    return (move_cost, servers_list)


# ── Adaptive Clustering ───────────────────────────────────────────────────────


def run_adaptive_clustering(path, window=20, reassign_every=10, **kwargs):
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for t, request in enumerate(requests):
        site_pos = clients[request]
        server = choose_server_adaptive_clustering(
            servers, site_pos, state, t=t, window=window, reassign_every=reassign_every, **kwargs
        )
        cost += manhattan(servers[server], site_pos)
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_adaptive_clustering(servers, site_pos, state, t=0, window=20, reassign_every=10, **kwargs):
    """
    Adaptive clustering: each server owns a zone defined by recent request history.

    Every `reassign_every` steps we recompute k centroids via k-means (5 iters)
    on the last `window` requests, then re-assign each server to the nearest
    centroid (greedy bipartite matching).  Between reassignments, each server
    greedily handles requests in its zone; ties fall back to nearest server.

    State keys:
      "history"        : sliding window of recent request positions
      "centroids"      : list of k centroid positions (or None before first fit)
      "zone_of_server" : zone_of_server[j] = centroid index owned by server j
    """
    k = len(servers)

    if "history" not in state:
        state["history"] = []
        state["centroids"] = [None] * k
        state["zone_of_server"] = list(range(k))

    state["history"].append(site_pos)
    if len(state["history"]) > window:
        state["history"].pop(0)

    # Periodically recompute zones
    if t % reassign_every == 0 and len(state["history"]) >= k:
        centroids = _kmeans(state["history"], k, n_iter=5, init_positions=servers)
        state["centroids"] = centroids
        state["zone_of_server"] = _greedy_assign(servers, centroids)

    centroids = state["centroids"]

    # Assign request to its nearest centroid, then serve with the owning server
    if any(c is not None for c in centroids):
        zone = min(range(k), key=lambda z: manhattan(centroids[z], site_pos) if centroids[z] is not None else math.inf)
        for j, z in enumerate(state["zone_of_server"]):
            if z == zone:
                return j

    # Fallback: nearest server
    return min(range(k), key=lambda j: manhattan(servers[j], site_pos))


def _kmeans(points, k, n_iter=5, init_positions=None):
    """
    K-means on Manhattan distance.
    Centroids are initialised from init_positions (server positions) if provided,
    otherwise spread evenly through the point list.
    """
    if init_positions is not None and len(init_positions) == k:
        centroids = list(init_positions)
    else:
        step = max(1, len(points) // k)
        centroids = [points[i * step] for i in range(k)]

    for _ in range(n_iter):
        clusters = [[] for _ in range(k)]
        for p in points:
            nearest = min(range(k), key=lambda z: manhattan(centroids[z], p))
            clusters[nearest].append(p)
        new_centroids = []
        for z in range(k):
            if clusters[z]:
                cx = sum(p[0] for p in clusters[z]) // len(clusters[z])
                cy = sum(p[1] for p in clusters[z]) // len(clusters[z])
                new_centroids.append((cx, cy))
            else:
                new_centroids.append(centroids[z])
        centroids = new_centroids

    return centroids


def _greedy_assign(servers, centroids):
    """
    Greedy bipartite assignment of servers to centroids by increasing distance.
    Returns zone_of_server where zone_of_server[j] is the centroid index for server j.
    """
    k = len(servers)
    pairs = sorted(
        (manhattan(servers[j], centroids[z]), j, z)
        for j in range(k) for z in range(k)
    )
    assigned_servers = set()
    assigned_zones = set()
    zone_of_server = [0] * k
    for _, j, z in pairs:
        if j not in assigned_servers and z not in assigned_zones:
            zone_of_server[j] = z
            assigned_servers.add(j)
            assigned_zones.add(z)
        if len(assigned_servers) == k:
            break
    return zone_of_server