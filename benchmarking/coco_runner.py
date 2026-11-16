"""Run a reproducible, resumable batch of COCO BBOB experiments.

Place this file at benchmarking/coco_runner.py in the optimization_algorithms
project, then run from the project root:
    python -m benchmarking.coco_runner

This is a custom CSV-based benchmarking runner, not the official COCO
Observer/cocopp recording workflow.
"""

import csv
import hashlib
import importlib
import json
import math
import random
import statistics
import time
import traceback
from collections import defaultdict
from pathlib import Path
import algorithms
import numpy as np



ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "algorithm", "function", "dimension", "instance", "run", "seed",
    "best_fitness", "evaluations", "seconds", "status", "error",
)


class EvaluationBudgetReached(Exception):
    """Raised before an evaluation that would exceed the budget."""


class BudgetedProblem:
    """Count and cap objective evaluations, regardless of optimizer type."""

    def __init__(self, problem, max_evaluations):
        self._problem = problem
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.best_fitness = float("inf")
        self.best_solution = None

    def evaluate(self, solution):
        if self.evaluations >= self.max_evaluations:
            raise EvaluationBudgetReached()

        fitness = float(self._problem.evaluate(solution))
        self.evaluations += 1

        if math.isfinite(fitness) and fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = np.asarray(solution, dtype=float).copy()

        return fitness

    def __getattr__(self, name):
        return getattr(self._problem, name)


def parse_functions(value):
    """Accept '1-24', '1,3,7-10', or [1, 3, 7]."""
    if isinstance(value, list):
        numbers = [int(item) for item in value]
    else:
        numbers = []
        for piece in str(value).split(","):
            piece = piece.strip()
            if "-" in piece:
                start, end = map(int, piece.split("-", 1))
                numbers.extend(range(start, end + 1))
            else:
                numbers.append(int(piece))
    result = sorted(set(numbers))
    if not result or not all(1 <= f <= 24 for f in result):
        raise ValueError("BBOB function indices must be in 1..24")
    return result


def make_seed(base_seed, algorithm, f, d, instance, run):
    # Stable across machines and Python processes, unlike hash(...).
    identifier = f"{base_seed}:{algorithm}:{f}:{d}:{instance}:{run}"
    return int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:4], "big")



def make_optimizer(algorithm, problem, algorithm_params):
    modules = {
        "ACO": (
            "aco.aco_optimizer",
            "ACO"
        ),
        "GeneticAlgorithm": (
            "genetic_algorithm.ga_optimizer",
            "GeneticAlgorithm"
        ),
    }

    if algorithm not in modules:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}"
        )

    module_name, class_name = modules[algorithm]

    optimizer_class = getattr(
        importlib.import_module(module_name),
        class_name
    )

    return optimizer_class(problem, **algorithm_params)



def summarize(results_file, summary_file):
    groups = defaultdict(list)
    with results_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["status"] not in ("finished", "budget_reached"):
                continue
            if not row["best_fitness"]:
                continue
            key = (
                row["algorithm"], row["function"],
                row["dimension"], row["instance"],
            )
            groups[key].append(row)

    with summary_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "algorithm", "function", "dimension", "instance", "runs",
            "mean_best_fitness", "std_best_fitness", "min_best_fitness",
            "mean_evaluations", "mean_seconds",
        ])
        for (alg, f, d, instance), rows in sorted(groups.items()):
            values = [float(row["best_fitness"]) for row in rows]
            writer.writerow([
                alg, f, d, instance, len(values),
                statistics.mean(values), statistics.pstdev(values),
                min(values),
                statistics.mean(int(row["evaluations"]) for row in rows),
                statistics.mean(float(row["seconds"]) for row in rows),
            ])


def main():
    with (ROOT / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)

    benchmark = config["coco_benchmark"]
    algorithms = benchmark.get("algorithms")
    if algorithms is None:
        algorithms = [config["run_settings"]["algorithm"]]
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    functions = parse_functions(benchmark["functions"])
    dimensions = [int(x) for x in benchmark["dimensions"]]
    instances = [int(x) for x in benchmark["instances"]]
    num_runs = int(benchmark["num_runs"])
    budget = int(benchmark["max_evaluations"])
    save_plots = bool(benchmark.get("save_plots", False))
    base_seed = benchmark.get("base_seed", 42)
    if base_seed is None:
        base_seed = 42

    if budget <= 0 or num_runs <= 0:
        raise ValueError("num_runs and max_evaluations must be positive")
    if not dimensions or not instances or not algorithms:
        raise ValueError("dimensions, instances and algorithms cannot be empty")

    suite = config.get("problem_params", {}).get("COCO", {}).get("suite", "bbob")
    if suite != "bbob":
        raise ValueError("This runner currently supports the 'bbob' suite only")

    selected_algorithm_params = {
        name: config["algorithm_params"][name] for name in algorithms
    }
    results_root = ROOT / "results" / "coco_benchmark"
    results_root.mkdir(parents=True, exist_ok=True)

    existing_tests = [
        p for p in results_root.iterdir()
        if p.is_dir() and p.name.startswith("test") and p.name[4:].isdigit()
    ]

    if existing_tests:
        next_number = max(
            int(p.name[4:])
            for p in existing_tests
        ) + 1
    else:
        next_number = 1

    results_root = ROOT / "results" / "coco_benchmark"
    results_root.mkdir(parents=True, exist_ok=True)

    existing_tests = [
        p for p in results_root.iterdir()
        if p.is_dir()
        and p.name.startswith("test")
        and p.name[4:].isdigit()
    ]

    if existing_tests:
        next_number = max(
            int(p.name[4:])
            for p in existing_tests
        ) + 1
    else:
        next_number = 1

    test_name = f"test{next_number}"

    output_dir = results_root / test_name
    output_dir.mkdir(parents=True, exist_ok=True)

    signature = {
        "benchmark": benchmark,
        "suite": suite,
        "algorithms": algorithms,
        "algorithm_params": selected_algorithm_params,
        "test_name": test_name,
    }

    (output_dir / "configuration.json").write_text(
        json.dumps(signature, indent=2),
        encoding="utf-8"
    )

    results_file = output_dir / "runs.csv"
    summary_file = output_dir / "summary.csv"
    
    completed = set()
    if results_file.exists():
        with results_file.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["status"] in ("finished", "budget_reached"):
                    completed.add((
                        row["algorithm"], int(row["function"]),
                        int(row["dimension"]), int(row["instance"]),
                        int(row["run"]),
                    ))

    create_coco_problem = importlib.import_module(
        "problems.coco"
    ).create_coco_problem

    total = (len(algorithms) * len(functions) * len(dimensions)
             * len(instances) * num_runs)
    print(f"COCO benchmark: {total} planned runs")
    print(f"Budget per run: {budget} objective evaluations")
    print(f"Results folder: {output_dir}")
    print(f"Already completed (resumable): {len(completed)}")

    with results_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if file.tell() == 0:
            writer.writeheader()
            file.flush()

        count = 0
        for algorithm in algorithms:
            for f in functions:
                for d in dimensions:
                    for instance in instances:
                        for run in range(1, num_runs + 1):
                            count += 1
                            key = (algorithm, f, d, instance, run)
                            if key in completed:
                                continue

                            seed = make_seed(base_seed, *key)
                            params = dict(selected_algorithm_params[algorithm])

                            if algorithm == "ACO":
                                params["seed"] = seed
                                params["max_evaluations"] = budget

                            elif algorithm == "GeneticAlgorithm":
                                # GA does not currently accept seed in __init__
                                random.seed(seed)
                                np.random.seed(seed)

                            raw_problem = None
                            monitored = None
                            optimizer = None
                            status = "finished"
                            error = ""
                            started = time.perf_counter()

                            try:
                                raw_problem = create_coco_problem(
                                    suite=suite, function=f,
                                    dimension=d, instance=instance,
                                )
                                monitored = BudgetedProblem(raw_problem, budget)
                                optimizer = make_optimizer(
                                    algorithm, monitored, params
                                )
                                try:
                                    optimizer.optimize()
                                except EvaluationBudgetReached:
                                    status = "budget_reached"

                                if monitored.evaluations == 0:
                                    raise RuntimeError(
                                        "Optimizer did not call problem.evaluate()"
                                    )
                                if monitored.best_solution is None:
                                    raise RuntimeError(
                                        "No finite objective value was obtained"
                                    )

                                if save_plots and getattr(optimizer, "history", None):
                                    from benchmarking.coco_plots import plot_convergence
                                    plot_convergence(
                                        optimizer.history,
                                        output_dir / "plots"
                                        / f"{algorithm}_f{f:03d}_d{d:02d}"
                                          f"_i{instance:02d}_r{run}.png",
                                        algorithm_name=algorithm,
                                        problem_name=f"bbob_f{f:03d}_i{instance:02d}_d{d:02d}",
                                    )

                            except Exception as exc:
                                status = "error"
                                error = f"{type(exc).__name__}: {exc}"
                                traceback.print_exc()
                            finally:
                                if raw_problem is not None:
                                    close = getattr(raw_problem, "close", None)
                                    if callable(close):
                                        close()

                            seconds = time.perf_counter() - started
                            row = {
                                "algorithm": algorithm,
                                "function": f,
                                "dimension": d,
                                "instance": instance,
                                "run": run,
                                "seed": seed,
                                "best_fitness": (
                                    monitored.best_fitness
                                    if monitored is not None
                                    and monitored.best_solution is not None
                                    else ""
                                ),
                                "evaluations": (
                                    monitored.evaluations
                                    if monitored is not None else 0
                                ),
                                "seconds": round(seconds, 4),
                                "status": status,
                                "error": error,
                            }
                            writer.writerow(row)
                            file.flush()  # Keep progress if interrupted
                            print(
                                f"[{count}/{total}] {algorithm} "
                                f"f={f} d={d} i={instance} run={run} "
                                f"fitness={row['best_fitness']} "
                                f"evals={row['evaluations']} [{status}]",
                                flush=True,
                            )

    summarize(results_file, summary_file)
    print(f"\nDone. Raw results: {results_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
