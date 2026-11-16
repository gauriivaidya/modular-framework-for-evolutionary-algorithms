from pathlib import Path
import numpy as np


class MDMKPProblem:
    def __init__(
        self,
        costs,
        A_le,
        b_le,
        A_ge,
        b_ge,
        name="mdmkp",
        penalty_factor=10000.0
    ):
        self.costs = np.asarray(costs, dtype=float)

        self.A_le = np.asarray(A_le, dtype=float)
        self.b_le = np.asarray(b_le, dtype=float)

        self.A_ge = np.asarray(A_ge, dtype=float)
        self.b_ge = np.asarray(b_ge, dtype=float)

        self.n_items = len(self.costs)

        self.size = self.n_items
        self.dimension = self.n_items
        self.num_dimensions = self.n_items

        # Framework minimizes
        self.is_minimization = True

        self.name = name
        self.penalty_factor = float(penalty_factor)

    def evaluate(self, chromosome):
        x = np.asarray(chromosome, dtype=int)

        objective = float(np.dot(self.costs, x))

        penalty = 0.0

        # <= constraint violations
        if len(self.A_le) > 0:
            lhs_le = self.A_le @ x
            violation_le = np.maximum(lhs_le - self.b_le, 0.0)
            penalty += float(np.sum(violation_le))

        # >= constraint violations
        if len(self.A_ge) > 0:
            lhs_ge = self.A_ge @ x
            violation_ge = np.maximum(self.b_ge - lhs_ge, 0.0)
            penalty += float(np.sum(violation_ge))

        return -objective + self.penalty_factor * penalty

    # -------------------------
    # ACO interface
    # -------------------------

    def initial_state(self):
        return []

    def feasible_moves(self, state):
        if len(state) >= self.n_items:
            return []

        # Both decisions are possible.
        # Constraint violations are handled by evaluate().
        return [0, 1]

    def apply_move(self, state, move):
        return state + [int(move)]

    def is_complete(self, state):
        return len(state) == self.n_items

    def finalize_solution(self, state):
        return list(state)

    def pheromone_key(self, state, move):
        item_idx = len(state)
        return item_idx, int(move)

    def all_components(self):
        return [
            (i, decision)
            for i in range(self.n_items)
            for decision in (0, 1)
        ]

    def solution_components(self, solution):
        return [
            (i, int(bit))
            for i, bit in enumerate(solution)
        ]

    def heuristic_value(self, state, move):
        item_idx = len(state)

        if item_idx >= self.n_items:
            return 1.0

        if int(move) == 0:
            return 1.0

        cost = self.costs[item_idx]

        # Must remain positive because ACO heuristic
        # is normally raised to beta.
        return max(abs(float(cost)), 1e-10)

    def solution_to_array(self, solution):
        return np.asarray(solution, dtype=int)

    def neighbors(self, solution):
        solution = list(solution)

        for i in range(self.n_items):
            neighbor = solution[:]
            neighbor[i] = 1 - neighbor[i]
            yield neighbor
            
            
def load_mdmkp_instance(
    filepath,
    test_problem=1,
    variant=1
):
    if not 1 <= variant <= 6:
        raise ValueError("variant must be between 1 and 6")

    tokens = Path(filepath).read_text().split()
    tokens = [int(x) for x in tokens]

    pos = 0

    K = tokens[pos]
    pos += 1

    if not 1 <= test_problem <= K:
        raise ValueError(
            f"test_problem must be between 1 and {K}"
        )

    for test_idx in range(1, K + 1):

        n = tokens[pos]
        m = tokens[pos + 1]
        pos += 2

        # m <= constraint vectors
        A_le = np.array(
            tokens[pos:pos + m * n],
            dtype=float
        ).reshape(m, n)

        pos += m * n

        # <= right-hand sides
        b_le = np.array(
            tokens[pos:pos + m],
            dtype=float
        )

        pos += m

        # m possible >= constraint vectors
        A_ge_all = np.array(
            tokens[pos:pos + m * n],
            dtype=float
        ).reshape(m, n)

        pos += m * n

        # >= right-hand sides
        b_ge_all = np.array(
            tokens[pos:pos + m],
            dtype=float
        )

        pos += m

        # 6 objective vectors
        costs_all = np.array(
            tokens[pos:pos + 6 * n],
            dtype=float
        ).reshape(6, n)

        pos += 6 * n

        if test_idx != test_problem:
            continue

        # Variants 1/4 -> q = 1
        # Variants 2/5 -> q = floor(m/2)
        # Variants 3/6 -> q = m
        if variant in (1, 4):
            q = 1

        elif variant in (2, 5):
            q = m // 2

        else:
            q = m

        costs = costs_all[variant - 1]

        A_ge = A_ge_all[:q]
        b_ge = b_ge_all[:q]

        return costs, A_le, b_le, A_ge, b_ge

    raise ValueError(
        f"Could not find test problem {test_problem}"
    )
    
    
def create_mdmkp_problem(
    instance,
    test_problem=1,
    variant=1,
    penalty_factor=10000.0
):
    root = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "or_library"
        / "mdmkp"
    )

    filepath = root / instance

    if not filepath.exists():
        raise FileNotFoundError(
            f"MDMKP instance not found: {filepath}"
        )

    costs, A_le, b_le, A_ge, b_ge = load_mdmkp_instance(
        filepath,
        test_problem=test_problem,
        variant=variant
    )

    return MDMKPProblem(
        costs=costs,
        A_le=A_le,
        b_le=b_le,
        A_ge=A_ge,
        b_ge=b_ge,
        penalty_factor=penalty_factor,
        name=f"{Path(instance).stem}-p{test_problem}-v{variant}"
    )