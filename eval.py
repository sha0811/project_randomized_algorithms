import math
from tqdm import tqdm
import sys
from utilities import instances_dir, results_dir
from algorithms import (
    run_greedy,
    run_balance,
    run_balance_aggressive,
    run_balance_exponential,
    run_balance_sqrt,
    run_balance_time_decay,
    run_balance_time_decay_inverse,
    run_balance_reuse_zero,
    run_site_affinity,
    run_double_coverage,
    run_wfa,
    run_adaptive_clustering,
    run_random_among_nearest,
    run_balance_random,
    run_bbmn_fractional,
    run_harmonic,
    run_softmin_balance,
)


def test_algo_all_instances(algorithm, algo_name="greedy", **algo_kw):
    """Tests the given algorithm on all instances and writes the results in the results file"""
    results_file = f"{algo_name}.txt"
    results_path = results_dir / results_file
    ratios = []
    with open(results_path, "w") as out:
        for file_path in sorted(instances_dir.glob("*.inst")):
            cost, opt = algorithm(file_path, **algo_kw)
            ratio = cost / opt
            ratios.append(ratio)
            line = f"{file_path.name}: cost={cost}, opt={opt}, ratio={ratio:.4f}\n"
            out.write(line)
        if ratios:
            n = len(ratios)
            mean_ratio = sum(ratios) / n
            var = sum((r - mean_ratio) ** 2 for r in ratios) / (n - 1) if n > 1 else 0.0
            std_err = math.sqrt(var / n)
            ci_half = 1.96 * std_err
            out.write(f"\nmean_ratio: {mean_ratio:.4f}  ci95: [{mean_ratio - ci_half:.4f}, {mean_ratio + ci_half:.4f}]\n")


def test_randomized_algo_all_instances(algorithm, algo_name, n_runs=10_000, **algo_kw):
    """For each instance, run the algorithm n_runs times and write mean ratio and 95% CI per instance"""
    results_path = results_dir / f"{algo_name}.txt"
    all_mean_ratios = []
    with open(results_path, "w") as out:
        i = 0
        for file_path in sorted(instances_dir.glob("*.inst")):
            # if i == 4: 
            #     break
            i += 1
            ratios = []
            for _ in tqdm(range(n_runs), desc=file_path.name, unit="run"):
                cost, opt = algorithm(file_path, **algo_kw)
                ratios.append(cost / opt)
            mean_ratio = sum(ratios) / n_runs
            var = sum((r - mean_ratio) ** 2 for r in ratios) / (n_runs - 1) if n_runs > 1 else 0
            std_err = math.sqrt(var / n_runs)
            ci_half = 1.96 * std_err
            all_mean_ratios.append(mean_ratio)
            out.write(
                f"{file_path.name}: mean_ratio={mean_ratio:.4f} ci95=[{mean_ratio - ci_half:.4f}, {mean_ratio + ci_half:.4f}]\n"
            )
        if all_mean_ratios:
            overall_mean = sum(all_mean_ratios) / len(all_mean_ratios)
            out.write(f"\noverall_mean_ratio: {overall_mean:.4f}\n")


# Mapping algo name to its properties
ALGORITHMS = {
    "greedy": (run_greedy, True, {}),
    "balance": (run_balance, True, {"alpha": 0.5}),
    "balance_aggressive": (run_balance_aggressive, True, {"alpha": 0.01}),
    "balance_exponential": (run_balance_exponential, True, {"alpha": 0.5, "beta": 0.01}),
    "balance_sqrt": (run_balance_sqrt, True, {"alpha": 0.5}),
    "balance_time_decay": (run_balance_time_decay, True, {"alpha_0": 0.55, "decay": 0.9988}),
    "balance_time_decay_inverse": (run_balance_time_decay_inverse, True, {"alpha_0": 0.6, "T": 8000}),
    "balance_reuse_zero": (run_balance_reuse_zero, True, {"alpha_0": 0.55, "decay": 0.999}),
    "site_affinity": (run_site_affinity, True, {"alpha": 0.5, "max_ratio": 1.4}),
    "double_coverage": (run_double_coverage, True, {}),
    "wfa": (run_wfa, True, {}),
    "adaptive_clustering": (run_adaptive_clustering, True, {"window": 20, "reassign_every": 10}),
    "random_among_nearest": (run_random_among_nearest, False, {"m": 2}),
    "balance_random": (run_balance_random, False, {"alpha": 0.5}),
    "harmonic": (run_harmonic, False, {}),
    "softmin_balance": (run_softmin_balance, False, {"alpha": 0.5, "temperature": 10.0}),
}


if __name__ == "__main__":
    # Deterministic algorithms
    # test_algo_all_instances(run_greedy, algo_name="greedy")
    # test_algo_all_instances(run_balance, algo_name="balance", alpha=0.5)
    # test_algo_all_instances(run_balance_aggressive, algo_name="balance_aggressive", alpha=0.01)
    # test_algo_all_instances(run_balance_exponential, algo_name="balance_exponential", alpha=0.5, beta=0.01)
    # test_algo_all_instances(run_double_coverage, algo_name="double_coverage")
    # # Randomized algorithms
    # test_randomized_algo_all_instances(
    #     run_random_among_nearest, algo_name="random_among_nearest", m=2
    # )
    # test_randomized_algo_all_instances(
    #     run_balance_random, algo_name="balance_random", alpha=0.5
    # )
    test_randomized_algo_all_instances(
        run_bbmn_fractional,
        algo_name="bbmn",          # results/bbmn.txt
        n_runs=50,             # same as the other randomized algos
        # hst_seed is intentionally omitted so each run gets a fresh random HST
        eps=0.05, deta=0.05
    )
# =======
#     if len(sys.argv) > 1:
#         # case where we want to run a specific algorithm
#         for name in sys.argv[1:]:
#             name = name.strip().lower()
#             if name not in ALGORITHMS:
#                 print(f"Unknown algorithm: {name}. Available: {', '.join(sorted(ALGORITHMS))}")
#                 sys.exit(1)
#             algo, deterministic, kwargs = ALGORITHMS[name]
#             if deterministic:
#                 test_algo_all_instances(algo, algo_name=name, **kwargs)
#             else:
#                 test_randomized_algo_all_instances(algo, algo_name=name, **kwargs)
#             print(f"Done: {name} -> results/{name}.txt")
#     else:
#         # Run all algorithms
#         for name, (algo, deterministic, kwargs) in ALGORITHMS.items():
#             if deterministic:
#                 test_algo_all_instances(algo, algo_name=name, **kwargs)
#             else:
#                 test_randomized_algo_all_instances(algo, algo_name=name, **kwargs)
#             print(f"Done: {name}")
# >>>>>>> 976cd1b763d5fac28d31d963761055692d0a8395
