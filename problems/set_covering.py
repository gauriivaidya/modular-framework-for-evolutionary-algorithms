from pathlib import Path
import numpy as np


class SetCoveringProblem:

    def __init__(
        self,
        costs,
        row_to_columns,
        name="set_covering",
        penalty_factor=10000.0
    ):
        self.costs = np.asarray(costs, dtype=float)

        self.row_to_columns = [
            list(columns)
            for columns in row_to_columns
        ]

        self.num_rows = len(self.row_to_columns)
        self.num_columns = len(self.costs)

        self.size = self.num_columns
        self.dimension = self.num_columns
        self.num_dimensions = self.num_columns

        self.name = name

        # We minimize total selected-set cost
        self.is_minimization = True

        self.penalty_factor = float(penalty_factor)

        # --------------------------------------------------
        # Pre-compute which rows each column covers
        # --------------------------------------------------

        self.column_to_rows = [
            []
            for _ in range(self.num_columns)
        ]

        for row, columns in enumerate(
            self.row_to_columns
        ):
            for col in columns:
                self.column_to_rows[col].append(row)

    # =====================================================
    # Objective function
    # =====================================================

    def evaluate(self, solution):

        x = np.asarray(
            solution,
            dtype=int
        )

        if len(x) != self.num_columns:
            raise ValueError(
                f"Expected solution length "
                f"{self.num_columns}, got {len(x)}"
            )

        total_cost = float(
            np.dot(
                self.costs,
                x
            )
        )

        uncovered_rows = 0

        for columns in self.row_to_columns:

            covered = any(
                x[col] == 1
                for col in columns
            )

            if not covered:
                uncovered_rows += 1

        penalty = (
            uncovered_rows
            * self.penalty_factor
        )

        return total_cost + penalty

    # =====================================================
    # ACO generic interface
    # =====================================================

    def initial_state(self):
        return []

    def feasible_moves(self, state):

        if len(state) >= self.num_columns:
            return []

        return [0, 1]

    def apply_move(
        self,
        state,
        move
    ):
        return state + [int(move)]

    def is_complete(
        self,
        state
    ):
        return (
            len(state)
            == self.num_columns
        )

    def finalize_solution(
        self,
        state
    ):
        return list(state)

    # -----------------------------------------------------
    # Pheromone
    # -----------------------------------------------------

    def pheromone_key(
        self,
        state,
        move
    ):
        column = len(state)

        return (
            column,
            int(move)
        )

    def all_components(self):

        return [
            (column, choice)
            for column
            in range(self.num_columns)
            for choice in (0, 1)
        ]

    def solution_components(
        self,
        solution
    ):

        return [
            (
                column,
                int(choice)
            )
            for column, choice
            in enumerate(solution)
        ]

    # -----------------------------------------------------
    # Heuristic
    # -----------------------------------------------------

    def heuristic_value(
        self,
        state,
        move
    ):
        column = len(state)

        if column >= self.num_columns:
            return 1.0

        # Skipping a column
        if int(move) == 0:
            return 1.0

        cost = self.costs[column]

        # Number of rows covered by this column
        coverage = len(
            self.column_to_rows[column]
        )

        if coverage == 0:
            return 1e-12

        # Cheap columns that cover many rows
        # are more attractive
        return max(
            coverage / max(cost, 1e-12),
            1e-12
        )

    # =====================================================
    # Generic framework helpers
    # =====================================================

    def solution_to_array(
        self,
        solution
    ):
        return np.asarray(
            solution,
            dtype=int
        )

    def neighbors(
        self,
        solution
    ):
        solution = list(solution)

        for i in range(
            self.num_columns
        ):

            neighbor = solution[:]

            neighbor[i] = (
                1 - neighbor[i]
            )

            yield neighbor


# =========================================================
# OR-Library loader
# =========================================================

def load_set_covering_instance(
    filepath
):

    filepath = Path(filepath)

    tokens = [
        int(x)
        for x in filepath.read_text().split()
    ]

    pos = 0

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    num_rows = tokens[pos]
    num_columns = tokens[pos + 1]

    pos += 2

    # -----------------------------------------------------
    # Column costs
    # -----------------------------------------------------

    costs = np.asarray(
        tokens[
            pos:
            pos + num_columns
        ],
        dtype=float
    )

    pos += num_columns

    # -----------------------------------------------------
    # Each row:
    #
    # k
    # col1 col2 ... colk
    #
    # OR-Library uses 1-based indices.
    # Convert them to Python's 0-based indexing.
    # -----------------------------------------------------

    row_to_columns = []

    for _ in range(
        num_rows
    ):

        num_covering_columns = (
            tokens[pos]
        )

        pos += 1

        columns = tokens[
            pos:
            pos + num_covering_columns
        ]

        pos += num_covering_columns

        columns = [
            column - 1
            for column in columns
        ]

        row_to_columns.append(
            columns
        )

    return (
        costs,
        row_to_columns
    )


# =========================================================
# Factory
# =========================================================

def create_set_covering_problem(
    instance,
    penalty_factor=10000.0
):

    root = (
        Path(__file__).resolve()
        .parent.parent
        / "data"
        / "or_library"
        / "set_covering"
    )

    filepath = root / instance

    if not filepath.exists():

        raise FileNotFoundError(
            f"Set Covering instance "
            f"not found: {filepath}"
        )

    (
        costs,
        row_to_columns
    ) = load_set_covering_instance(
        filepath
    )

    return SetCoveringProblem(
        costs=costs,
        row_to_columns=row_to_columns,
        name=filepath.stem,
        penalty_factor=penalty_factor
    )