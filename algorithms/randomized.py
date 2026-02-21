import random
from utilities import load_instance, manhattan


def run_random_among_nearest(path, m=2, **kwargs):
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        site_pos = clients[request]
        server = choose_server_random_among_nearest(servers, site_pos, state, m=m, **kwargs)
        cost += manhattan(servers[server], site_pos)
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_random_among_nearest(servers, site_pos, state, m=2, **kwargs):
    """Among the m nearest servers, pick one at random. m=2 by default."""
    if "rng" not in state:
        state["rng"] = random.Random()
    rng = state["rng"]
    n = len(servers)
    # to ensure that m is not greater than the number of servers
    m = min(m, n)
    dists = [(manhattan(servers[j], site_pos), j) for j in range(n)]
    dists.sort(key=lambda x: x[0])
    candidates = [dists[i][1] for i in range(m)]
    return rng.choice(candidates)


def run_balance_random(path, alpha=0.5, epsilon=1e-6, **kwargs):
    instance = load_instance(path)
    clients = instance["sites"]
    requests = instance["requests"]
    k = instance["k"]
    servers = [(0, 0)] * k
    state = {}
    cost = 0
    for request in requests:
        site_pos = clients[request]
        server = choose_server_balance_random(servers, site_pos, state, alpha=alpha, epsilon=epsilon, **kwargs)
        cost += manhattan(servers[server], site_pos)
        servers[server] = site_pos
    return (cost, instance["opt"])


def choose_server_balance_random(servers, site_pos, state, alpha=0.5, epsilon=1e-6, **kwargs):
    """
    Same as Balance but with random handling of equality: when several servers have (nearly) the same score,
    pick one uniformly at random.
    """
    if "rng" not in state:
        state["rng"] = random.Random()
    if "total_dist" not in state:
        state["total_dist"] = [0] * len(servers)
    rng = state["rng"]
    scores = [
        manhattan(servers[j], site_pos) + alpha * state["total_dist"][j]
        for j in range(len(servers))
    ]
    best_score = min(scores)
    candidates = [j for j in range(len(servers)) if scores[j] <= best_score + epsilon]
    best_server = rng.choice(candidates)
    d = manhattan(servers[best_server], site_pos)
    state["total_dist"][best_server] += d
    return best_server
