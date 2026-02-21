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



