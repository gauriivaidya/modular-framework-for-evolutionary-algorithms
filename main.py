import csv
import json
import importlib
import os
import re
import time
import numpy as np
from datetime import datetime

from core.logger_setup import setup_logger


def to_snake_case(name):
    """Converts PascalCase/camelCase strings to snake_case."""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def main():
    logger = setup_logger()

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    try:
        with open("config.json", "r") as f:
            config = json.load(f)

        logger.info("Configuration file loaded successfully.")

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error with config.json: {e}")
        return

    num_runs = config["run_settings"].get("num_runs", 1)

    summary_data = []

    best_fitnesses = []
    execution_times = []
    convergence_generations = []

    best_overall_fitness = float("inf")

    results_dir = "results"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    experiment_start_time = time.perf_counter()

    logger.info(
        f"--- Starting Experiment: {num_runs} runs ---"
    )

    # --------------------------------------------------
    # Algorithm mappings
    # --------------------------------------------------

    alg_aliases = {
        "aco": "ACO",
        "genetic_algorithm": "GeneticAlgorithm",
        "geneticalgorithm": "GeneticAlgorithm"
    }

    alg_file_aliases = {
        "ACO": "aco_optimizer",
        "GeneticAlgorithm": "genetic_algorithm"
    }

    # --------------------------------------------------
    # Runs
    # --------------------------------------------------

    for i in range(num_runs):

        run_id = (
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            f"_{i + 1}"
        )

        logger.info(
            f"--- Starting Run {i + 1}/{num_runs} "
            f"(ID: {run_id}) ---"
        )

        # --------------------------------------------------
        # Load algorithm and problem
        # --------------------------------------------------

        try:
            alg_key = config["run_settings"]["algorithm"]

            alg_name = alg_aliases.get(
                alg_key.lower(),
                alg_key
            )

            prob_name = config["run_settings"]["problem"]

            alg_filename = alg_file_aliases.get(
                alg_name,
                to_snake_case(alg_name)
            )

            prob_filename = to_snake_case(
                prob_name
            )

            # ----------------------------------------------
            # Load algorithm
            # ----------------------------------------------

            try:
                alg_module = importlib.import_module(
                    f"algorithms.{alg_filename}"
                )

            except ImportError:
                alg_module = importlib.import_module(
                    f"aco.{alg_filename}"
                )

            AlgClass = getattr(
                alg_module,
                alg_name
            )

            # ----------------------------------------------
            # Load problem
            # ----------------------------------------------

            prob_module = importlib.import_module(
                f"problems.{prob_filename}"
            )

            prob_params = (
                config
                .get("problem_params", {})
                .get(prob_name, {})
            )

            create_func_name = (
                f"create_{prob_filename}_problem"
            )

            if hasattr(
                prob_module,
                create_func_name
            ):
                create_func = getattr(
                    prob_module,
                    create_func_name
                )

                problem_instance = create_func(
                    **prob_params
                )

            else:
                instance_name = (
                    f"{prob_filename}_problem"
                )

                problem_instance = getattr(
                    prob_module,
                    instance_name
                )

            logger.info(
                f"Algorithm: {alg_name}, "
                f"Problem: {problem_instance.name}"
            )

        except (
            ImportError,
            AttributeError,
            TypeError,
            FileNotFoundError,
            ValueError
        ) as e:

            logger.exception(
                f"Failed to load algorithm or problem: {e}"
            )

            return

        # --------------------------------------------------
        # Load algorithm parameters
        # --------------------------------------------------

        alg_params = (
            config
            .get("algorithm_params", {})
            .get(alg_name, {})
        )

        try:
            optimizer = AlgClass(
                problem_instance,
                **alg_params
            )

        except TypeError as e:
            logger.exception(
                f"Failed to initialize {alg_name}: {e}"
            )
            return
        
        # --------------------------------------------------
        # Run optimizer
        # --------------------------------------------------

        run_start_time = time.perf_counter()

        try:
            best_position, best_fitness = (
                optimizer.optimize()
                
            )

        except Exception as e:
            logger.exception(
                f"Optimization failed: {e}"
            )
            return
        
        
        # --------------------------------------------------
        # COCO convergence graph
        # --------------------------------------------------

        if config["run_settings"]["problem"] == "COCO":

            from pathlib import Path
            from uuid import uuid4
            from benchmarking.coco_plots import plot_convergence

            algorithm_name = config["run_settings"]["algorithm"]
            problem_name = problem_instance.name

            plot_convergence(
                history=optimizer.history,
                output_path=(
                    Path("results/coco")
                    / f"{algorithm_name}_{problem_name}_{uuid4().hex[:8]}.png"
                ),
                algorithm_name=algorithm_name,
                problem_name=problem_name
            )

    



        run_end_time = time.perf_counter()

        execution_time = (
            run_end_time - run_start_time
        )

        logger.info(
            f"--- Run {i + 1} Finished in "
            f"{execution_time:.2f}s ---"
        )

        logger.info(
            f"  Best Fitness: {best_fitness}"
        )

        logger.info(
            f"  Found at generation: "
            f"{optimizer.convergence_generation}"
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        best_fitnesses.append(
            best_fitness
        )

        execution_times.append(
            execution_time
        )

        if (
            optimizer.convergence_generation
            is not None
        ):
            convergence_generations.append(
                optimizer.convergence_generation
            )

        if best_fitness < best_overall_fitness:
            best_overall_fitness = best_fitness

        # --------------------------------------------------
        # Run configuration
        # --------------------------------------------------

        run_config = {
            "algorithm": alg_name,
            "problem": prob_name,
            "algorithm_params": alg_params,
            "problem_params": prob_params
        }

        # --------------------------------------------------
        # Convert best position
        # --------------------------------------------------

        if isinstance(
            best_position,
            np.ndarray
        ):
            best_position_serializable = (
                best_position.tolist()
            )
        else:
            best_position_serializable = list(
                best_position
            )

        # --------------------------------------------------
        # JSON results
        # --------------------------------------------------

        results = {
            "run_id": run_id,

            "problem_name":
                problem_instance.name,

            "algorithm_name":
                alg_name,

            "best_fitness":
                float(best_fitness),

            "best_position":
                best_position_serializable,

            "convergence_generation":
                optimizer.convergence_generation,

            "execution_time_seconds":
                execution_time,

            "run_configuration":
                run_config,

            "history":
                optimizer.history
        }

        json_filename = os.path.join(
            results_dir,
            f"results_{run_id}.json"
        )

        with open(
            json_filename,
            "w"
        ) as f:

            json.dump(
                results,
                f,
                default=lambda o:
                    o.tolist()
                    if isinstance(o, np.ndarray)
                    else o,
                indent=4
            )

        # --------------------------------------------------
        # Analytics CSV
        # --------------------------------------------------

        csv_filename = os.path.join(
            results_dir,
            f"analytics_{run_id}.csv"
        )

        try:
            with open(
                csv_filename,
                "w",
                newline=""
            ) as csvfile:

                writer = csv.writer(
                    csvfile
                )

                writer.writerow([
                    "run_id",
                    "generation",
                    "fitness",
                    "solution"
                ])

                for entry in optimizer.history:

                    solution = entry.get(
                        "solution",
                        []
                    )

                    solution_str = ";".join(
                        map(
                            str,
                            solution
                        )
                    )

                    writer.writerow([
                        run_id,
                        entry.get(
                            "generation"
                        ),
                        entry.get(
                            "fitness"
                        ),
                        solution_str
                    ])

        except IOError as e:
            logger.error(
                f"Failed to write analytics CSV "
                f"for run {run_id}: {e}"
            )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        summary_data.append({
            "execution_id":
                run_id,

            "execution_time":
                f"{execution_time:.4f}",

            "best_fitness":
                best_fitness,

            "parameters":
                json.dumps(
                    alg_params
                ),

            "convergence":
                optimizer.convergence_generation
        })

    # --------------------------------------------------
    # Aggregate statistics
    # --------------------------------------------------

    aggregate_stats = {}

    if best_fitnesses:

        aggregate_stats = {
            "mean_best_fitness":
                float(
                    np.mean(
                        best_fitnesses
                    )
                ),

            "std_best_fitness":
                float(
                    np.std(
                        best_fitnesses
                    )
                ),

            "best_overall_fitness":
                float(
                    best_overall_fitness
                ),

            "mean_execution_time":
                float(
                    np.mean(
                        execution_times
                    )
                ),

            "mean_convergence_generation":
                (
                    float(
                        np.mean(
                            convergence_generations
                        )
                    )
                    if convergence_generations
                    else None
                )
        }

        logger.info(
            "--- Aggregate Statistics ---"
        )

        logger.info(
            f"  Mean Best Fitness: "
            f"{aggregate_stats['mean_best_fitness']:.6f}"
        )

        logger.info(
            f"  Std Best Fitness: "
            f"{aggregate_stats['std_best_fitness']:.6f}"
        )

        logger.info(
            f"  Best Overall Fitness: "
            f"{aggregate_stats['best_overall_fitness']:.6f}"
        )

        logger.info(
            f"  Mean Execution Time: "
            f"{aggregate_stats['mean_execution_time']:.6f}"
        )

        logger.info(
            f"  Mean Convergence Generation: "
            f"{aggregate_stats['mean_convergence_generation']}"
        )

    # --------------------------------------------------
    # Summary CSV
    # --------------------------------------------------

    summary_filename = os.path.join(
        results_dir,
        (
            "summary_log_"
            f"{datetime.now().strftime('%Y-%m-%d')}.csv"
        )
    )

    try:
        if summary_data:

            with open(
                summary_filename,
                "w",
                newline=""
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=summary_data[0].keys()
                )

                writer.writeheader()

                writer.writerows(
                    summary_data
                )

            logger.info(
                f"Summary log saved to "
                f"{summary_filename}"
            )

    except (IOError, IndexError) as e:
        logger.error(
            f"Failed to write summary log: {e}"
        )

    # --------------------------------------------------
    # Aggregate CSV
    # --------------------------------------------------

    aggregate_filename = os.path.join(
        results_dir,
        (
            "aggregate_summary_"
            f"{datetime.now().strftime('%Y-%m-%d')}.csv"
        )
    )

    try:
        if aggregate_stats:

            with open(
                aggregate_filename,
                "w",
                newline=""
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=
                        aggregate_stats.keys()
                )

                writer.writeheader()

                writer.writerow(
                    aggregate_stats
                )

            logger.info(
                f"Aggregate metrics saved to "
                f"{aggregate_filename}"
            )

    except IOError as e:
        logger.error(
            f"Failed to write aggregate metrics: {e}"
        )

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------

    experiment_end_time = (
        time.perf_counter()
    )

    logger.info(
        f"--- Experiment Finished in "
        f"{experiment_end_time - experiment_start_time:.2f}s ---"
    )


if __name__ == "__main__":
    main()