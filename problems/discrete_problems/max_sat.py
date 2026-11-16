"""
MAX-SAT Problem:
Given a Boolean formula in Conjunctive Normal Form (CNF),
find a variable assignment that maximises the number of satisfied clauses.

Representation : binary chromosome  [0/1]  length = num_variables
Evaluation cost: O(num_clauses) = O(D) at standard ratio ~4.3 clauses/variable
Is_minimization: False  (we maximise satisfied clauses)

Instance generation follows the standard random 3-SAT model:
  - Each clause picks 3 distinct variables uniformly at random
  - Each variable is negated with probability 0.5
  - Phase-transition ratio: clauses_per_var ≈ 4.267 (hardest instances)
"""

import numpy as np


class MAXSATProblem:
    """
    MAX-SAT: maximise the number of satisfied clauses in a 3-CNF formula.

    Attributes
    ----------
    num_variables  : int   — chromosome length / problem dimension
    num_clauses    : int   — number of clauses in the formula
    clauses        : list  — each clause is a list of 3 literals;
                             positive int  →  variable (1-indexed)
                             negative int  →  negated variable
    """

    def __init__(self, num_variables, clauses, name=None):
        self.num_variables  = int(num_variables)
        self.clauses        = clauses          # list of list-of-3 literals
        self.num_clauses    = len(clauses)

        # GA interface — mirrors TSP / OneMax
        self.size           = self.num_variables
        self.dimension      = self.num_variables
        self.num_dimensions = self.num_variables
        self.is_minimization = False           # maximise satisfied clauses
        self.name = name or f"maxsat-{self.num_variables}v-{self.num_clauses}c"

    # ------------------------------------------------------------------
    def evaluate(self, chromosome):
        """
        Count the number of satisfied clauses.

        Parameters
        ----------
        chromosome : array-like of 0/1, length = num_variables

        Returns
        -------
        float  — number of satisfied clauses  (higher is better)

        Complexity: O(num_clauses * 3) = O(D)
        """
        satisfied = 0
        for clause in self.clauses:
            for literal in clause:
                var_idx = abs(literal) - 1          # 0-indexed
                val     = int(chromosome[var_idx])
                # literal > 0  → satisfied if var == 1
                # literal < 0  → satisfied if var == 0
                if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                    satisfied += 1
                    break                           # clause satisfied, move on
        return float(satisfied)

    # ------------------------------------------------------------------
    def satisfaction_ratio(self, chromosome):
        """Fraction of clauses satisfied — useful for logging."""
        return self.evaluate(chromosome) / self.num_clauses


# ──────────────────────────────────────────────────────────────────────
# FACTORY
# ──────────────────────────────────────────────────────────────────────

def create_max_sat_problem(
    num_variables   = 50,
    clauses_per_var = 4.267,          # phase-transition ratio for hardest instances
    literals_per_clause = 3,          # standard 3-SAT
    seed            = 42,
    name            = None,
):
    """
    Generate a random k-SAT instance.

    Parameters
    ----------
    num_variables       : int   — problem dimension D
    clauses_per_var     : float — ratio clauses/variables (4.267 = hardest 3-SAT)
    literals_per_clause : int   — k in k-SAT (default 3)
    seed                : int   — RNG seed for reproducibility
    name                : str   — optional instance name

    Returns
    -------
    MAXSATProblem
    """
    rng         = np.random.default_rng(seed)
    num_clauses = int(round(num_variables * clauses_per_var))

    clauses = []
    for _ in range(num_clauses):
        # Pick k distinct variables (1-indexed)
        vars_chosen = rng.choice(num_variables, size=literals_per_clause, replace=False) + 1
        # Negate each with probability 0.5
        signs       = rng.choice([-1, 1], size=literals_per_clause)
        clause      = list(vars_chosen * signs)
        clauses.append(clause)

    instance_name = name or f"maxsat-{num_variables}v-{num_clauses}c-s{seed}"
    return MAXSATProblem(num_variables, clauses, name=instance_name)


# ──────────────────────────────────────────────────────────────────────
# LOAD FROM DIMACS .cnf FILE  (SATLIB format)
# ──────────────────────────────────────────────────────────────────────

def load_dimacs(filepath):
    """
    Load a SATLIB / DIMACS CNF file.

    File format:
        c  comment lines
        p cnf <num_variables> <num_clauses>
        <lit1> <lit2> <lit3> 0   ← one clause per line, terminated by 0

    Returns
    -------
    MAXSATProblem
    """
    clauses      = []
    num_variables = None
    num_clauses   = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                # p cnf <vars> <clauses>
                parts         = line.split()
                num_variables = int(parts[2])
                num_clauses   = int(parts[3])
                continue
            # Clause line: integers terminated by 0
            literals = [int(x) for x in line.split() if x != '0']
            if literals:
                clauses.append(literals)

    if num_variables is None:
        raise ValueError(f"No 'p cnf' header found in {filepath}")

    name = filepath.split('/')[-1].replace('.cnf', '')
    return MAXSATProblem(num_variables, clauses, name=name)


# ──────────────────────────────────────────────────────────────────────
# DEFAULT INSTANCE
# ──────────────────────────────────────────────────────────────────────

max_sat_problem = create_max_sat_problem()