"""
Permutation Flowshop Scheduling Problem (PFSP):
Given n jobs and m machines, find a job permutation that minimises makespan.
All jobs visit machines in the same order (machines 0 → m-1).

Representation : permutation chromosome  [0..n_jobs-1]
Evaluation cost: O(n_jobs * n_machines) = O(D) where D = n_jobs
Is_minimization: True  (minimise makespan)

Benchmark instances: Taillard (1993) — the gold standard for PFSP.
  ta001–ta010 :  20 jobs × 5 machines
  ta011–ta020 :  20 jobs × 10 machines
  ta021–ta030 :  50 jobs × 5 machines
  ta031–ta040 :  50 jobs × 10 machines
  ta041–ta050 : 100 jobs × 5 machines
  ta051–ta060 : 100 jobs × 10 machines
  ta061–ta070 : 200 jobs × 10 machines
  ta071–ta080 : 200 jobs × 20 machines
  ta081–ta090 : 500 jobs × 20 machines
  ta091–ta100 : 500 jobs × 20 machines (different seeds)

Reference:
  Taillard, E. (1993). Benchmarks for basic scheduling problems.
  European Journal of Operational Research, 64(2), 278-285.
"""

import numpy as np


class FlowshopProblem:
    """
    Permutation Flowshop Scheduling Problem.

    Attributes
    ----------
    n_jobs     : int              — chromosome length / problem dimension D
    n_machines : int              — number of machines (structural parameter)
    proc_times : np.ndarray       — processing times, shape (n_jobs, n_machines)
    """

    def __init__(self, n_jobs, n_machines, proc_times, name=None):
        self.n_jobs     = int(n_jobs)
        self.n_machines = int(n_machines)
        self.proc_times = np.asarray(proc_times, dtype=float)

        if self.proc_times.shape != (self.n_jobs, self.n_machines):
            raise ValueError(
                f"proc_times must be shape ({self.n_jobs}, {self.n_machines}), "
                f"got {self.proc_times.shape}"
            )

        # GA interface — mirrors TSP
        self.size            = self.n_jobs
        self.dimension       = self.n_jobs
        self.num_dimensions  = self.n_jobs
        self.is_minimization = True
        self.name = name or f"flowshop-{self.n_jobs}j-{self.n_machines}m"

    # ------------------------------------------------------------------
    def evaluate(self, chromosome):
        """
        Compute makespan for a given job permutation.

        Uses the standard DP completion-time recurrence:
            C[j][m] = proc_times[job][m]
                    + max(C[j-1][m], C[j][m-1])

        Complexity: O(n_jobs * n_machines) = O(D * n_machines)
        Since n_machines is fixed per experiment, this is O(D).

        Parameters
        ----------
        chromosome : array-like, permutation of [0 .. n_jobs-1]

        Returns
        -------
        float — makespan (lower is better)
        """
        job_order = [int(c) for c in chromosome]
        n, m      = self.n_jobs, self.n_machines
        p         = self.proc_times

        # C[i][k] = completion time of i-th job in sequence on machine k
        C = np.zeros((n, m))

        for i, job in enumerate(job_order):
            for k in range(m):
                finish_prev_job      = C[i - 1][k] if i > 0 else 0.0
                finish_prev_machine  = C[i][k - 1] if k > 0 else 0.0
                C[i][k] = p[job][k] + max(finish_prev_job, finish_prev_machine)

        return float(C[n - 1][m - 1])   # makespan = last job, last machine

    # ------------------------------------------------------------------
    def evaluate_fast(self, chromosome):
        """
        Vectorised makespan — same result, faster for large instances.
        Uses np.maximum.accumulate along both axes.
        """
        job_order = [int(c) for c in chromosome]
        p         = self.proc_times[job_order, :]    # reorder rows

        C = np.zeros_like(p)
        C[0, :] = np.cumsum(p[0, :])                # first job
        C[:, 0] = np.cumsum(p[:, 0])                # first machine

        for i in range(1, self.n_jobs):
            for k in range(1, self.n_machines):
                C[i, k] = p[i, k] + max(C[i - 1, k], C[i, k - 1])

        return float(C[-1, -1])


# ──────────────────────────────────────────────────────────────────────
# FACTORY — random instance
# ──────────────────────────────────────────────────────────────────────

def create_flowshop_problem(
    n_jobs     = 20,
    n_machines = 5,
    seed       = 42,
    name       = None,
):
    """
    Generate a random PFSP instance with processing times in [1, 99].
    Matches Taillard's generation scheme.

    Parameters
    ----------
    n_jobs     : int — problem dimension D
    n_machines : int — fixed structural parameter (typically 5 or 10)
    seed       : int — RNG seed for reproducibility
    name       : str — optional instance name

    Returns
    -------
    FlowshopProblem
    """
    rng        = np.random.default_rng(seed)
    proc_times = rng.integers(1, 100, size=(n_jobs, n_machines))
    instance_name = name or f"flowshop-{n_jobs}j-{n_machines}m-s{seed}"
    return FlowshopProblem(n_jobs, n_machines, proc_times, name=instance_name)


# ──────────────────────────────────────────────────────────────────────
# LOAD TAILLARD INSTANCE  (.txt format)
# ──────────────────────────────────────────────────────────────────────

def load_taillard(filepath):
    """
    Load a Taillard benchmark instance.

    Taillard file format:
        number of jobs, number of machines, initial seed, upper bound, lower bound
        processing times: n_jobs rows × n_machines columns
        (may contain multiple instances separated by blank lines)

    Returns
    -------
    list of FlowshopProblem  — one per instance block in the file
    """
    instances = []
    instance_idx = 0

    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        # Header line: n_jobs n_machines seed ub lb
        header = lines[i].split()
        if len(header) < 2:
            i += 1
            continue

        try:
            n_jobs     = int(header[0])
            n_machines = int(header[1])
        except ValueError:
            i += 1
            continue

        i += 1
        proc_times = []

        # Read n_jobs rows of processing times
        while len(proc_times) < n_jobs and i < len(lines):
            row = [int(x) for x in lines[i].split()]
            proc_times.append(row)
            i += 1

        if len(proc_times) == n_jobs:
            name = f"taillard-{instance_idx:03d}-{n_jobs}j-{n_machines}m"
            instances.append(
                FlowshopProblem(n_jobs, n_machines, proc_times, name=name)
            )
            instance_idx += 1

    return instances


# ──────────────────────────────────────────────────────────────────────
# STANDARD INSTANCE SETS  (D sweep for complexity study)
# ──────────────────────────────────────────────────────────────────────

def create_instance_set(
    n_jobs_list = [20, 50, 100, 200],
    n_machines  = 5,
    n_instances = 5,
    base_seed   = 0,
):
    """
    Create a set of instances for a D-sweep experiment.
    For each n_jobs value, generates n_instances with different seeds.

    Returns
    -------
    dict: {n_jobs: [FlowshopProblem, ...]}
    """
    instance_set = {}
    for n_jobs in n_jobs_list:
        instance_set[n_jobs] = [
            create_flowshop_problem(
                n_jobs     = n_jobs,
                n_machines = n_machines,
                seed       = base_seed + i,
                name       = f"flowshop-{n_jobs}j-{n_machines}m-inst{i}",
            )
            for i in range(n_instances)
        ]
    return instance_set


# ──────────────────────────────────────────────────────────────────────
# DEFAULT INSTANCE
# ──────────────────────────────────────────────────────────────────────

flowshop_problem = create_flowshop_problem()