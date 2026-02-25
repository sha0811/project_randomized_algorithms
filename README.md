# k-Server Project

Game theory project (M1): online algorithms for the k-server problem, with evaluation on provided instances and comparison to the offline optimal cost.

## Problem

- **k servers** (technicians) process **requests** that arrive in sequence.
- Each request must be served **immediately** by one server (no queuing).
- Movement is **instantaneous**; **distance** is the **Manhattan** (L1) norm.
- **Goal**: minimise the sum of distances travelled by all servers over the whole sequence.
- Sites lie on the grid **[0, 99]²**; all servers start at **(0, 0)**.

The **competitive ratio** of an online algorithm is (algorithm cost) / (offline optimal cost). The provided instances include the offline optimal value (`# opt`).

## Project structure

```
Projet/
├── README.md
├── utilities.py          # Instance loading, Manhattan distance, paths (instances, results)
├── eval.py               # Entry point: run all algorithms on all instances
├── algorithms/
│   ├── __init__.py       # Exports (run_* and choose_server_*)
│   ├── deterministic.py  # Deterministic algorithms (greedy, balance, double coverage, etc.)
│   └── randomized.py     # Randomized algorithms (random among nearest, balance random)
├── k-server_instances/   # .inst files (one instance per file)
└── results/              # .txt result files (one per algorithm)
```

- Each algorithm exposes a **`run_*(path, **kwargs)`** function that takes an instance path and returns **`(cost, opt)`**. The evaluator only uses this interface.
- **`choose_server_*`** functions are the server-selection policies used inside the `run_*` functions.

## Requirements

- Python 3
- `tqdm`: `pip install tqdm`

## Running the evaluation

From the project root:

```bash
python eval.py
```

- **Deterministic** algorithms are run once per instance; results are written to `results/<algo_name>.txt` with cost, opt, and ratio per instance, then **mean ratio** and **95% confidence interval** (over instances) at the end of the file.
- **Randomized** algorithms are run **10,000 times** per instance; each result file contains **mean ratio** and **95% CI** per instance, then the overall mean ratio.

## Instance format (`.inst`)

Each file contains:

- `# opt` then the offline optimal cost
- `# k` then the number of servers
- `# sites` then one line per site: `x y` (coordinates)
- `# demandes` then one line of integers: site indices requested (request sequence)

Sites are numbered in order of appearance (starting from 0).

## Implemented algorithms

| Algorithm           | Type        | Short description |
|--------------------|-------------|-------------------|
| Greedy             | Deterministic | Nearest server |
| Balance            | Deterministic | Minimise distance + α × (distance already travelled by server) |
| Balance time decay | Deterministic | Balance with time-decaying α (decay=0.9988) |
| Balance reuse zero  | Deterministic | Like balance time decay but when a server is already at the request, prefer the one that last served this site (best mean ratio) |
| Balance aggressive | Deterministic | Balance with quadratic penalty on load |
| Balance exponential| Deterministic | Balance with exponential penalty on load |
| Double Coverage    | Deterministic | Two nearest servers move toward the request until one reaches it |
| Random among nearest | Randomized | Pick uniformly at random among the m nearest servers |
| Balance random     | Randomized | Balance with random tie-breaking when scores are equal |

Parameters (α, β, m, etc.) are set in the calls to `test_algo_all_instances` / `test_randomized_algo_all_instances` in `eval.py`.

## Results

- **Deterministic**: `results/<algo_name>.txt` — one line per instance (cost, opt, ratio) + a line with `mean_ratio` and `ci95`.
- **Randomized**: `results/<algo_name>.txt` — one line per instance (mean_ratio, ci95) + a line with `overall_mean_ratio`.

To compare algorithms, compare **mean ratios** and **CIs** across these files.
