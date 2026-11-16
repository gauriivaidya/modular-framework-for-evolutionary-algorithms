from pathlib import Path
import numpy as np
    

class BinPackingProblem:

    def __init__(
        self,
        item_sizes,
        capacity,
        best_known=None,
        name="bin_packing"
    ):
        self.item_sizes = np.asarray(
            item_sizes,
            dtype=int
        )

        self.capacity = int(capacity)
        self.best_known = best_known

        self.num_items = len(self.item_sizes)

        self.size = self.num_items
        self.dimension = self.num_items
        self.num_dimensions = self.num_items

        self.name = name
        self.is_minimization = True

    # =====================================================
    # Objective
    # =====================================================

    def evaluate(self, solution):
        """
        solution[i] = bin index assigned to item i

        Fitness = number of bins used.
        """

        solution = np.asarray(
            solution,
            dtype=int
        )

        if len(solution) != self.num_items:
            raise ValueError(
                f"Expected {self.num_items} assignments, "
                f"got {len(solution)}"
            )

        if self.num_items == 0:
            return 0.0

        if np.any(solution < 0):
            return float("inf")

        used_bins = (
            int(np.max(solution)) + 1
        )

        loads = np.zeros(
            used_bins,
            dtype=int
        )

        for item, bin_index in enumerate(
            solution
        ):
            loads[bin_index] += (
                self.item_sizes[item]
            )

        # Reject infeasible assignments
        if np.any(loads > self.capacity):
            return float("inf")

        return float(used_bins)

    # =====================================================
    # ACO state
    # =====================================================

    def initial_state(self):
        """
        State is a list of bin assignments.
        state[i] = bin assigned to item i.
        """
        return []

    def feasible_moves(self, state):
        """
        For the next item, return existing feasible bins
        plus one possible new bin.
        """

        item_index = len(state)

        if item_index >= self.num_items:
            return []

        item_size = self.item_sizes[
            item_index
        ]

        # Current loads
        if len(state) == 0:
            return [0]

        num_used_bins = (
            max(state) + 1
        )

        loads = [
            0
            for _ in range(num_used_bins)
        ]

        for i, bin_index in enumerate(
            state
        ):
            loads[bin_index] += (
                self.item_sizes[i]
            )

        moves = []

        # Existing bins
        for bin_index in range(
            num_used_bins
        ):
            if (
                loads[bin_index]
                + item_size
                <= self.capacity
            ):
                moves.append(
                    bin_index
                )

        # Always permit opening exactly one new bin
        moves.append(
            num_used_bins
        )

        return moves

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
            == self.num_items
        )

    def finalize_solution(
        self,
        state
    ):
        return list(state)

    # =====================================================
    # Pheromone
    # =====================================================

    def pheromone_key(
        self,
        state,
        move
    ):
        """
        Pheromone based on item -> bin assignment.
        """
        item_index = len(state)

        return (
            item_index,
            int(move)
        )

    def all_components(self):
        """
        In worst case every item gets its own bin.
        """

        return [
            (item, bin_index)
            for item in range(
                self.num_items
            )
            for bin_index in range(
                self.num_items
            )
        ]

    def solution_components(
        self,
        solution
    ):
        return [
            (
                item_index,
                int(bin_index)
            )
            for item_index, bin_index
            in enumerate(solution)
        ]

    # =====================================================
    # Heuristic
    # =====================================================

    def heuristic_value(
        self,
        state,
        move
    ):
        """
        Prefer placing items into bins that become fuller.

        This is similar to a best-fit heuristic.
        """

        item_index = len(state)

        if item_index >= self.num_items:
            return 1.0

        item_size = self.item_sizes[
            item_index
        ]

        # First item
        if len(state) == 0:
            return 1.0

        num_used_bins = (
            max(state) + 1
        )

        # Opening a new bin
        if move >= num_used_bins:
            remaining = (
                self.capacity
                - item_size
            )

        else:
            load = 0

            for i, bin_index in enumerate(
                state
            ):
                if bin_index == move:
                    load += (
                        self.item_sizes[i]
                    )

            remaining = (
                self.capacity
                - load
                - item_size
            )

        # Smaller remaining space = better
        return 1.0 / (
            remaining + 1.0
        )

    # =====================================================
    # Framework helpers
    # =====================================================

    def solution_to_array(
        self,
        solution
    ):
        return np.asarray(
            solution,
            dtype=int
        )


# =========================================================
# OR-Library loader
# =========================================================

def load_bin_packing_instance(
    filepath,
    test_problem=1
):
    filepath = Path(filepath)

    lines = [
        line.strip()
        for line in filepath.read_text().splitlines()
        if line.strip()
    ]

    pos = 0

    num_problems = int(
        lines[pos]
    )

    pos += 1

    if not (
        1
        <= test_problem
        <= num_problems
    ):
        raise ValueError(
            f"test_problem must be between "
            f"1 and {num_problems}"
        )

    for problem_index in range(
        1,
        num_problems + 1
    ):

        identifier = lines[pos]
        pos += 1

        header = lines[pos].split()
        pos += 1

        capacity = int(
            header[0]
        )

        num_items = int(
            header[1]
        )

        best_known = int(
            header[2]
        )

        item_sizes = []

        for _ in range(
            num_items
        ):
            item_sizes.append(
                int(lines[pos])
            )
            pos += 1

        if problem_index == test_problem:
            return (
                identifier,
                capacity,
                best_known,
                item_sizes
            )

    raise ValueError(
        "Could not load requested test problem."
    )


# =========================================================
# Factory
# =========================================================

def create_bin_packing_problem(
    instance,
    test_problem=1
):

    root = (
        Path(__file__).resolve()
        .parent.parent
        / "data"
        / "or_library"
        / "bin_packing"
    )

    filepath = root / instance

    if not filepath.exists():
        raise FileNotFoundError(
            f"Bin Packing instance "
            f"not found: {filepath}"
        )

    (
        identifier,
        capacity,
        best_known,
        item_sizes
    ) = load_bin_packing_instance(
        filepath,
        test_problem=test_problem
    )

    return BinPackingProblem(
        item_sizes=item_sizes,
        capacity=capacity,
        best_known=best_known,
        name=identifier
    )