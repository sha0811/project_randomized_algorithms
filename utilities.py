from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
instances_dir = PROJECT_ROOT / "k-server_instances"
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)


def load_instance(path):
    """Returns a dictionary of the form  {"opt": opt, "k": k, "sites": sites, "requests": requests}
    from input file"""
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    i = 0
    while i < len(lines) and lines[i] != "# opt":
        i += 1
    i += 1
    opt = int(lines[i])
    while i < len(lines) and lines[i] != "# k":
        i += 1
    i += 1
    k = int(lines[i])
    while i < len(lines) and lines[i] != "# sites":
        i += 1
    i += 1
    sites = []
    while i < len(lines) and lines[i] != "# demandes":
        if lines[i]:
            x, y = map(int, lines[i].split())
            sites.append((x, y))
        i += 1
    i += 1
    if i < len(lines) and lines[i]:
        requests = list(map(int, lines[i].split()))
    else:
        requests = []
    return {"opt": opt, "k": k, "sites": sites, "requests": requests}


def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def move_toward(p, q, d):
    """From p, move by Manhattan distance d toward q. Returns new position (or q if d >= manhattan(p,q))."""
    M = manhattan(p, q)
    if d >= M:
        return q
    dx = min(d, abs(q[0] - p[0]))
    dy = min(d - dx, abs(q[1] - p[1]))
    sign_x = 1 if q[0] >= p[0] else -1
    sign_y = 1 if q[1] >= p[1] else -1
    return (p[0] + sign_x * dx, p[1] + sign_y * dy)
