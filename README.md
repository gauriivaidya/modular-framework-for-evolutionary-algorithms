# Optimization Algorithms

A Python framework for implementing, testing, and comparing optimization algorithms across multiple problem types.

The project is intended for experimentation, benchmarking, and studying the behavior of optimization techniques under consistent evaluation conditions.

## Overview

This project contains implementations of optimization algorithms such as:

- Ant Colony Optimization
- Genetic Algorithms
- Optimization Algorithms

The goal of the project is to provide a clean structure for experimenting with algorithms, testing them on different problems, and comparing their performance.

## Project Structure

```text
OPTIMIZATION_ALGORITHMS/
├── aco/                    # Ant Colony Optimization implementations
│   ├── construction/       # Move-selection rules (modular engine)
│   ├── heuristic/          # Heuristic / visibility functions
│   ├── pheromone/          # Pheromone-update building blocks
│   └── variants/           # High-level update strategies
├── algorithms/             # Shared/general optimization algorithms
├── core/                   # Core interfaces, base classes, utilities
├── benchmarking/           # Or_Library instances and COCO code.
├── genetic_algorithm/      # Genetic Algorithm implementations
├── problems/               # Problem definitions and objective functions
├── results/                # Output files, plots, logs, and experiment results
├── sources/                # References, notes, or external source material
├── venv/                   # Local virtual environment, not committed
├── config.json             # Configuration file for experiments
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

# Requirements

* Python 3.10 or later
* `Git`

## Installation

Clone the repository:

```bash
git clone https://github.com/Adnan-Adda/optimization_algorithms.git
cd optimization_algorithms
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

macOS / Linux:

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

# Usage

The optimization framework is fully configurable through the `config.json` file.  The framework is fully configured through `config.json`. Algorithms, problems, and their parameters can be changed without modifying the source code.

## Running an Experiment

Run the framework from the project root:

```bash
python main.py
```

## Running COCO Benchmarking test

Run the COCO benchmarking test with in root:
python -m benchmarking.coco_runner

the benchmark settings are in config.json:

```json
  "coco_benchmark": {
    "functions": "1-2",
    "dimensions": [5,10],
    "instances": [1, 2],
    "num_runs": 2,
    "max_evaluations": 10000,
    "base_seed": 42,
    "save_plots": false
  },
```

You will find the relative docummentation here: [coco-platform.org/testsuites/bbob/overview.html](https://coco-platform.org/testsuites/bbob/overview.html)


The framework will:

- Load the selected algorithm and problem from `config.json`
- Execute the specified number of independent runs
- Save detailed results to the `results/` directory
- Generate summary statistics across all runs

## Selecting an Algorithm

Algorithms are selected inside the `run_settings` section.

Example:

```json
"run_settings": {
    "algorithm": "mmas",
    "problem": "tsp",
    "num_runs": 3
  },
```

## Configuring Algorithm Parameters

Each algorithm has its own parameter block inside `algorithm_params`. Feel free to change the values to your needs.

Example:

```json
    "MMAS": {
        "num_ants": 40,
        "num_generations": 150,
        "alpha": 1.0,
        "beta": 2.0,
        "evaporation_rate": 0.3,
        "q": 100.0,
        "initial_pheromone": 1.0,
        "tau_min": 0.01,
        "tau_max": 10.0,
        "use_global_best": true,
        "seed": null,
        "convergence_tol": 1e-12,
        "patience": 30
  },
```

# Results

After each experiment, the framework automatically generates several files inside the `results/` directory. These files allow both individual runs and complete experiments to be analyzed.

```text
results/
├── coco/graph.png
├── coco_benchmark/test<int>/ which includes configuration.json + runs.csv + summary.csv
├── aggregate_summary_<date>.csv
├── analytics_<timestamp>.csv
├── results_<timestamp>.json
├── run_<timestamp>.log
└── summary_log_<date>.csv
```



## coco_benchmark/test1/

Contains the complete results of a single COCO benchmark experiment.
This directory includes:

- configuration.json — Contains the exact configuration used for the experiment, including the benchmark functions, dimensions, instances, number of runs, evaluation limit, random seed, algorithm, and algorithm parameters.   configuration
- runs.csv — Contains the results from each individual algorithm run, including the performance measurements recorded during the benchmark.
- summary.csv — Contains aggregated performance statistics for the experiment, summarizing the results across runs.
  The experiment uses the BBOB suite with the GeneticAlgorithm and two runs per configuration.

### aggregate_summary.csv

Contains statistical information aggregated across all runs of an experiment.

This file includes metrics such as:

- Mean best fitness
- Standard deviation of the best fitness
- Best overall fitness
- Mean execution time
- Mean convergence generation

### analytics.csv

Stores generation-by-generation data for a single execution.

Each row corresponds to one generation and includes:

- Run identifier
- Generation number
- Best fitness found so far
- Best solution found so far

### results.json

Contains the complete results of a single execution.

The JSON file stores:

- Run identifier
- Algorithm name
- Problem name
- Best solution found
- Best fitness
- Execution time
- Convergence generation
- Algorithm parameters
- Problem parameters
- Complete optimization history

### run.log

Contains a chronological log of the experiment execution.

The log records information such as:

- Configuration loading
- Selected algorithm and problem
- Start and end of each run
- Execution times
- Best fitness values
- Aggregate statistics
- Warnings and error messages

# Algorithms

### aco

The `aco/` directory contains two complementary ACO designs:

1. **Matrix-based variants** — a base `ACO` class (`aco.py`) plus subclasses that
   work directly on a TSP distance matrix, all exposed via `__init__.py`.
2. **A modular engine** (`aco_optimizer.py`) — a problem-agnostic ACO assembled
   from interchangeable components loaded at runtime from the `construction/`,
   `heuristic/`, `pheromone/`, and `variants/` sub-packages.

> **Note:** The `"ACO"` block in `config.json` configures the modular engine,
> which `main.py` runs under the key `aco`. The matrix-based variants are selected
> by their exact class name and use their own parameter blocks.

#### Matrix-based variants

- `aco.py` — Base ACO: constructs routes probabilistically and applies evaporation plus optional elitist reinforcement. All variants below inherit from it.
- `adaptive_aco.py` — Linearly interpolates `alpha`, `beta`, and the evaporation rate across generations to shift from exploration to exploitation.
- `ant_colony_system.py` — Ant Colony System (ACS) with pseudo-random proportional selection (`q0`) and local pheromone updates during construction.
- `as_rank.py` — Rank-based Ant System where the best ants deposit pheromone weighted by rank, with extra global-best reinforcement.
- `beam_aco.py` — Combines ACO with beam search, keeping a beam of the best partial routes and expanding them stochastically or greedily.
- `bwas.py` — Best-Worst Ant System; rewards the iteration-best ant, penalizes the worst, and diversifies via pheromone mutation or restart on stagnation.
- `elitist_aco.py` — ACO with stronger elitist reinforcement of the global-best solution.
- `focused_aco.py` — Partially destroys and reinserts the best route (with candidate lists) and applies 2-opt local search.
- `heterogeneous_aco.py` — Uses several ant types with different `alpha`/`beta`/`q0` settings to balance exploration and exploitation.
- `hypercube_aco.py` — Hyper-Cube Framework ACO that keeps pheromone values within a normalized bounded range.
- `max_min_aco.py` — Max-Min Ant System; clips pheromone to `[tau_min, tau_max]` and by default deposits only along the best route.
- `mmas.py` — Alternative Max-Min Ant System; initializes pheromone at `tau_max` and deposits via the global- or iteration-best route.
- `multi_colony_aco.py` — Runs several independent colonies that periodically exchange information through migration.
- `parallel_aco.py` — Advances multiple colonies in parallel, sharing the global best via periodic pheromone migration.
- `population_aco.py` — Maintains a set of high-quality solutions and rebuilds the pheromone matrix from them each generation.
- `rank_based_aco.py` — Rank-Based Ant System where top-ranked ants deposit pheromone weighted by rank, plus global-best reinforcement.
- `restart_aco.py` — Restarts the pheromone matrix after a number of non-improving generations, optionally preserving the best solution.
- `smoothing_aco.py` — Periodically smooths pheromone values toward a target to reduce stagnation.

#### Modular engine (`aco_optimizer.py`)

A problem-agnostic ACO that selects its behavior at runtime from four sub-packages:

- `construction/` — move-selection rules: `probabilistic` (roulette wheel), `greedy`, `epsilon_greedy`, `candidate_list`, and `pseudo_random_proportional` (ACS-style).
- `heuristic/` — visibility functions: `problem` (delegates to the problem), `inverse_cost`, `value_density`, `randomized`, and `uniform`.
- `pheromone/` — update building blocks: `deposit`, `evaporation`, `bounds`, `max_min`, `local_update`, and `quality`.
- `variants/` — high-level update strategies: `standard`, `elitist`, `acs`, `mmas`, and `rank_based`.

### algorithms

The `algorithms/` directory contains implementations of various nature inspired, swarm based, and evolutionary optimization algorithms.

- `aco.py` —  Adapter file for main.py.
- `african_buffalo_optimization.py` — Inspired by buffalo herd movement and communication.
- `animal_migration.py` — Inspired by animal migration and population movement.
- `artificial_bee_colony.py` — Based on the foraging behavior of honey bees.
- `bat_algorithm.py` — Uses echolocation-inspired search and movement.
- `bird_swarm.py` — Inspired by bird foraging, vigilance, and flight behavior.
- `chicken_swarm_optimization.py` — Based on the hierarchical behavior of chicken groups.
- `coyote_optimization.py` — Inspired by social interaction within coyote packs.
- `genetic_algorithm.py` — Adapter file for main.py.
- `hierarchical_swarm.py` — Swarm-based optimization using a hierarchical population structure.
- `jumping_frogs_optimization.py` — Inspired by shuffled frog-leaping and cooperative search.
- `pigeon_inspired_optimization.py` — Based on pigeon navigation and homing behavior.
- `pso.py` — Particle Swarm Optimization using personal and global best solutions.
- `sailfish_optimizer.py` — Inspired by cooperative hunting behavior of sailfish.
- `sec_sse_partridge_optimization.py` — Partridge-inspired optimization using exploration and exploitation strategies.
- `social_spider_optimization.py` — Based on cooperative behavior within spider colonies.
- `spider_monkey_optimization.py` — Inspired by the social structure of spider monkeys.
- `termite_colony.py` — Inspired by collective termite behavior.
- `weightless_swarm.py` — Swarm optimization using simplified movement and update mechanisms.
- `whale_optimization.py` — Inspired by humpback whale bubble-net hunting behavior.

### genetic_algorithm

The `genetic_algorithm/` directory contains the core Genetic Algorithm implementation together with crossover, mutation, representation, and selection operators.

#### Crossover

The `crossover/` directory contains different crossover operators used to generate offspring from parent solutions.

- `cmx.py` — Implements the CMX crossover operator.
- `cx.py` — Implements Cycle Crossover (CX).
- `mpox.py` — Implements the MPOX crossover operator.
- `mppmx.py` — Implements the MPPMX crossover operator.
- `ox.py` — Implements Order Crossover (OX).
- `pmx.py` — Implements Partially Mapped Crossover (PMX).
- `rbx.py` — Implements the RBX crossover operator.
- `twopx.py` — Implements two-point crossover.

#### Mutation

The `mutation/` directory contains operators used to introduce variation into individuals.

- `inversion.py` — Reverses a selected section of a chromosome.
- `scramble.py` — Randomly rearranges genes within a selected section.
- `swap.py` — Swaps the positions of two genes.

#### Representation

The `representation/` directory defines different ways solutions can be encoded.

- `binary.py` — Represents solutions using binary chromosomes.
- `integer.py` — Represents solutions using integer-valued chromosomes.
- `permutation.py` — Represents solutions as permutations, commonly used for ordering problems such as TSP.

#### Selection

The `selection/` directory contains methods for selecting individuals for reproduction.

- `bts.py` — Implements the BTS selection method.
- `eebs.py` — Implements the EEBS selection method.
- `nts.py` — Implements the NTS selection method.
- `rank.py` — Selects individuals based on their fitness ranking.
- `roulette.py` — Implements roulette-wheel fitness-proportionate selection.
- `srs.py` — Implements the SRS selection method.
- `tournament.py` — Selects individuals through tournament-based competition.

#### Core

- `ga_optimizer.py` — Main Genetic Algorithm optimizer responsible for running the evolutionary process.
- `individual.py` — Defines an individual solution, including its chromosome and fitness information.

#### Benchmarking

The `benchmarking/` directory contains benchmark datasets, experiment runners, and visualization tools for evaluating and comparing optimization algorithms.

- `or_library/` — Contains OR-Library benchmark instances, organized by problem type:
  - `bin_packing/` — Benchmark instances for bin packing problems.
  - `graph_coloring/` — Benchmark instances for graph coloring problems.
  - `job_shop/` — Benchmark instances for job shop scheduling problems.
  - `mdmkp/` — Benchmark instances for multidimensional multiple-choice knapsack problems.
  - `set_covering/` — Benchmark instances for set covering problems.
  - `vrp/` — Benchmark instances for vehicle routing problems.
- `coco_plots.py` — Generates convergence graphs to visualize optimization performance.
- `coco_runner.py` — Executes automated COCO benchmark experiments across multiple functions, dimensions, instances, and algorithms, saving results for performance comparison.
