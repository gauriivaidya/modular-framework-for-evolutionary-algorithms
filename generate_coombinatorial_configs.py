"""
Unified config generator for combinatorial GA runtime complexity study.
Generates configs for all 6 problems across full operator grid.

Problems:
  Binary     : OneMax, GeneralKnapsack, MAXSAT
  Permutation: TSP, Flowshop, PMedian

Fixed budget : P × G = 40,000
Instances    : 5 per problem (different seeds)
Runs         : 10 per config
Total configs: 40,500  (6 problems × 5 instances × 1,350 operator configs)
Total runs   : 405,000
"""

import json
import random
import os
import csv
from itertools import product

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
RUNS_PER_CONFIG = 10
EVAL_BUDGET     = 40_000
OUTPUT_DIR      = "combinatorial_configs"
MASTER_CSV      = os.path.join(OUTPUT_DIR, "config_index.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(99)   # reproducible rate sampling

# ─────────────────────────────────────────────
# P×G PAIRS  (P × G = 40,000 exactly)
# ─────────────────────────────────────────────
PG_PAIRS = [
    (p, EVAL_BUDGET // p)
    for p in range(50, 601, 50)
    if EVAL_BUDGET % p == 0 and EVAL_BUDGET // p >= 20
]
# [(50,800),(100,400),(200,200),(250,160),(400,100),(500,80)]

# ─────────────────────────────────────────────
# GRID AXES  (shared across all problems)
# ─────────────────────────────────────────────
MUTATION_METHODS  = ["inversion", "scramble", "swap"]
SELECTION_METHODS = ["tournament", "roulette", "rank"]
TOURNAMENT_SIZES  = [2, 3, 5, 7, 10]
ELITISM_VALUES    = [0, 1, 2, 3, 5]

CROSSOVER_RATE_RANGE = (0.5, 1.0)   # sampled uniformly
MUTATION_RATE_RANGE  = (0.001, 0.5)

# ─────────────────────────────────────────────
# PROBLEM DEFINITIONS
# ─────────────────────────────────────────────
# Each problem defines:
#   problem_str  : string used in "problem" field
#   params_key   : key used in "problem_params"
#   representation: "binary" or "permutation"
#   crossover    : "twopx" or "ox"
#   d_values     : list of dimension (D) values to sweep
#   instances    : list of seed values (5 per problem)
#   param_fn     : callable(d, seed) -> dict of problem_params

INSTANCES = list(range(5))   # seeds 0–4

PROBLEMS = [
    # ── Binary ──────────────────────────────────────────────────────
    {
        "problem_str":    "discrete_problems.OneMax",
        "params_key":     "OneMax",
        "representation": "binary",
        "crossover":      "twopx",
        "d_values":       [10, 20, 50, 100, 200],
        "instances":      INSTANCES,
        "param_fn":       lambda d, seed: {
            "size": d
            # OneMax is deterministic — seed unused but kept for consistency
        },
    },
    {
        "problem_str":    "discrete_problems.GeneralKnapsack",
        "params_key":     "GeneralKnapsack",
        "representation": "binary",
        "crossover":      "twopx",
        "d_values":       [10, 20, 50, 100, 200],
        "instances":      INSTANCES,
        "param_fn":       lambda d, seed: {
            "num_items": d,
            "capacity":  d * 10,    # scales with D
            "max_count": 3,         # general knapsack: max 3 of each item
            "seed":      seed,
        },
    },
    {
        "problem_str":    "discrete_problems.MAXSAT",
        "params_key":     "MAXSAT",
        "representation": "binary",
        "crossover":      "twopx",
        "d_values":       [20, 50, 100, 200],
        "instances":      INSTANCES,
        "param_fn":       lambda d, seed: {
            "num_variables":    d,
            "clauses_per_var":  4.267,   # phase-transition ratio
            "seed":             seed,
        },
    },
    # ── Permutation ─────────────────────────────────────────────────
    {
        "problem_str":    "discrete_problems.TSP",
        "params_key":     "TSP",
        "representation": "permutation",
        "crossover":      "ox",
        "d_values":       [20, 50, 100, 200],
        "instances":      INSTANCES,
        "param_fn":       lambda d, seed: {
            "n_cities": d,
            "seed":     seed,
        },
    },
    {
        "problem_str":    "discrete_problems.Flowshop",
        "params_key":     "Flowshop",
        "representation": "permutation",
        "crossover":      "ox",
        "d_values":       [20, 50, 100, 200],
        "instances":      INSTANCES,
        "param_fn":       lambda d, seed: {
            "n_jobs":     d,
            "n_machines": 5,         # fixed structural parameter
            "seed":       seed,
        },
    },
    {
        "problem_str":    "discrete_problems.PMedian",
        "params_key":     "PMedian",
        "representation": "binary",
        "crossover":      "twopx",
        "d_values":       [20, 50, 100, 200],
        "instances":      INSTANCES,
        "param_fn":       lambda d, seed: {
            "n_locations": d,
            "p":           max(2, d // 10),  # P = D/10, keeps eval O(D)
            "seed":        seed,
        },
    },
]

# ─────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────
config_counter = 0
rows           = []

all_operator_combos = list(product(
    MUTATION_METHODS,
    SELECTION_METHODS,
    PG_PAIRS,
    TOURNAMENT_SIZES,
    ELITISM_VALUES,
))

print(f"Operator grid per (problem × D × instance): {len(all_operator_combos)} configs")
print(f"Problems: {len(PROBLEMS)}")

total_configs = sum(
    len(prob["d_values"]) * len(prob["instances"]) * len(all_operator_combos)
    for prob in PROBLEMS
)
print(f"Total configs : {total_configs}")
print(f"Total GA runs : {total_configs * RUNS_PER_CONFIG}\n")

for prob_def in PROBLEMS:
    prob_str    = prob_def["problem_str"]
    params_key  = prob_def["params_key"]
    rep         = prob_def["representation"]
    crossover   = prob_def["crossover"]
    d_values    = prob_def["d_values"]
    instances   = prob_def["instances"]
    param_fn    = prob_def["param_fn"]

    prob_dir = os.path.join(OUTPUT_DIR, params_key.lower())
    os.makedirs(prob_dir, exist_ok=True)

    for d, seed in product(d_values, instances):

        for mut, sel, (pop, gen), t_size, elitism in all_operator_combos:

            # Sample rates once per config
            cx_rate  = round(random.uniform(*CROSSOVER_RATE_RANGE), 3)
            mut_rate = round(random.uniform(*MUTATION_RATE_RANGE),  3)

            # Cap tournament size at population size
            eff_t_size = min(t_size, pop)

            config = {
                "run_settings": {
                    "algorithm": "GeneticAlgorithm",
                    "problem":   prob_str,
                    "num_runs":  RUNS_PER_CONFIG,
                },
                "algorithm_params": {
                    "GeneticAlgorithm": {
                        "population_size":  pop,
                        "num_generations":  gen,
                        "crossover_method": crossover,
                        "mutation_method":  mut,
                        "selection_method": sel,
                        "representation":   rep,
                        "crossover_rate":   cx_rate,
                        "mutation_rate":    mut_rate,
                        "tournament_size":  eff_t_size,
                        "elitism":          elitism,
                        "multi_parent":     False,
                        "num_parents":      2,
                    }
                },
                "problem_params": {
                    params_key: param_fn(d, seed)
                },
                "_meta": {
                    "config_id":        config_counter,
                    "problem":          params_key,
                    "representation":   rep,
                    "dimension":        d,
                    "instance_seed":    seed,
                    "eval_budget":      pop * gen,
                    "population_size":  pop,
                    "num_generations":  gen,
                    "mutation_method":  mut,
                    "selection_method": sel,
                    "tournament_size":  eff_t_size,
                    "elitism":          elitism,
                    "crossover_rate":   cx_rate,
                    "mutation_rate":    mut_rate,
                },
            }

            filename = f"{params_key.lower()}_d{d:03d}_s{seed}_{config_counter:06d}.json"
            filepath = os.path.join(prob_dir, filename)

            with open(filepath, "w") as f:
                json.dump(config, f, indent=2)

            rows.append({
                "config_id":        config_counter,
                "filename":         os.path.join(params_key.lower(), filename),
                "problem":          params_key,
                "representation":   rep,
                "dimension":        d,
                "instance_seed":    seed,
                "eval_budget":      pop * gen,
                "population_size":  pop,
                "num_generations":  gen,
                "crossover_method": crossover,
                "mutation_method":  mut,
                "selection_method": sel,
                "crossover_rate":   cx_rate,
                "mutation_rate":    mut_rate,
                "tournament_size":  eff_t_size,
                "elitism":          elitism,
                "num_runs":         RUNS_PER_CONFIG,
            })

            config_counter += 1

    print(f"  {params_key:>15} : {len(d_values)} D-values × {len(instances)} instances "
          f"× {len(all_operator_combos)} configs = "
          f"{len(d_values)*len(instances)*len(all_operator_combos)} configs generated")

# ─────────────────────────────────────────────
# SAVE MASTER INDEX CSV
# ─────────────────────────────────────────────
headers = [
    "config_id", "filename", "problem", "representation",
    "dimension", "instance_seed", "eval_budget",
    "population_size", "num_generations",
    "crossover_method", "mutation_method", "selection_method",
    "crossover_rate", "mutation_rate",
    "tournament_size", "elitism", "num_runs",
]

with open(MASTER_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  Total configs generated : {config_counter}")
print(f"  Total GA runs           : {config_counter * RUNS_PER_CONFIG}")
print(f"  Master index saved      : {MASTER_CSV}")
print(f"{'─'*55}")
print(f"\nConfig breakdown by problem:")
from collections import Counter
prob_counts = Counter(r["problem"] for r in rows)
for prob, count in prob_counts.items():
    print(f"  {prob:>15} : {count:>6} configs  ({count*RUNS_PER_CONFIG:>7} runs)")
print(f"\nEval budget verification:")
budgets = set(r["eval_budget"] for r in rows)
print(f"  Unique budgets: {budgets}  ({'✅ all 40,000' if budgets == {40000} else '❌ MISMATCH'})")