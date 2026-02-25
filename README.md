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
project_randomized_algorithms/
├── README.md
├── utilities.py          # Instance loading, Manhattan distance, paths (instances, results)
├── eval.py               # Entry point: run all algorithms on all instances
├── algorithms/
│   ├── __init__.py       # Exports (run_* and choose_server_*)
│   ├── deterministic.py # Deterministic algorithms (greedy, balance, WFA, double coverage, etc.)
│   └── randomized.py     # Randomized algorithms (harmonic, softmin, random among nearest, etc.)
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

To run only specific algorithms:

```bash
python eval.py greedy balance wfa
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

| Algorithm                  | Type         | Short description |
|----------------------------|--------------|-------------------|
| Greedy                     | Deterministic | Nearest server |
| Balance                    | Deterministic | Minimise distance + α × (distance already travelled by server) |
| Balance aggressive         | Deterministic | Balance with quadratic penalty on load (α=0.01) |
| Balance exponential        | Deterministic | Balance with exponential penalty on load |
| Balance sqrt               | Deterministic | Balance with sqrt penalty on total distance |
| Balance time decay         | Deterministic | Balance with time-decaying α (α₀=0.55, decay=0.9988) |
| Balance time decay inverse | Deterministic | Balance with inverse time decay α₀/(1+t/T) (T=8000) |
| Balance reuse zero         | Deterministic | Like balance time decay; when cost is 0, prefer server that last served this site |
| Site affinity              | Deterministic | Prefer reusing the server that last served this site when within max_ratio×min_dist |
| Double coverage            | Deterministic | Two nearest servers move toward the request until one reaches it |
| WFA                        | Deterministic | Work function algorithm (pruned to max_configs configurations) |
| Adaptive clustering        | Deterministic | k-means on recent requests; servers assigned to zones, reassigned periodically |
| Random among nearest       | Randomized   | Pick uniformly at random among the m nearest servers (m=2) |
| Balance random             | Randomized   | Balance with random tie-breaking when scores are equal |
| Harmonic                   | Randomized   | Choose server with probability ∝ 1/distance (O(log k)-competitive) |
| Softmin balance            | Randomized   | Choose with probability ∝ exp(−score/temperature); score = distance + α×total_dist |

Parameters (α, β, m, temperature, etc.) are set in `eval.py` in the `ALGORITHMS` mapping.

## Results

- **Deterministic**: `results/<algo_name>.txt` — one line per instance (cost, opt, ratio) + a line with `mean_ratio` and `ci95`.
- **Randomized**: `results/<algo_name>.txt` — one line per instance (mean_ratio, ci95) + a line with `overall_mean_ratio`.

To compare algorithms, compare **mean ratios** and **CIs** across these files.
