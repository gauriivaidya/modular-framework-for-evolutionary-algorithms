import numpy as np


class KnapsackProblem:
    """
    0/1 Knapsack

    Representation:
      - solution/chromosome is a binary vector of length n_items
      - 1 = take item
      - 0 = do not take item

    Objective:
      - maximize total value
      - framework minimizes fitness, so evaluate() returns negative total value
    """

    def __init__(self, weights, values, capacity, name="knapsack"):
        self.weights = np.asarray(weights, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.capacity = float(capacity)

        if len(self.weights) != len(self.values):
            raise ValueError("weights and values must have the same length")

        self.n_items = len(self.weights)

        # Compatibility with your framework / GA-style metadata
        self.size = self.n_items
        self.dimension = self.n_items
        self.num_dimensions = self.n_items
        self.is_minimization = True
        self.name = name

    # ----------------------------
    # Objective
    # ----------------------------
    def evaluate(self, chromosome):
        solution = np.asarray(chromosome, dtype=int)

        if len(solution) != self.n_items:
            raise ValueError(f"solution length must be {self.n_items}")

        total_weight = float(np.dot(solution, self.weights))
        total_value = float(np.dot(solution, self.values))

        # Heavy penalty for infeasible solutions
        if total_weight > self.capacity:
            penalty = 1000.0 * (total_weight - self.capacity)
            return penalty - total_value

        # Minimize negative profit
        return -total_value

    # ----------------------------
    # ACO construction interface
    # ----------------------------
    def initial_state(self):
        return []

    def feasible_moves(self, state):
        item_idx = len(state)
        if item_idx >= self.n_items:
            return []

        current_weight = self._state_weight(state)

        # Always allowed to skip item
        moves = [0]

        # Allowed to take item only if feasible
        if current_weight + self.weights[item_idx] <= self.capacity:
            moves.append(1)

        return moves

    def apply_move(self, state, move):
        return state + [int(move)]

    def is_complete(self, state):
        return len(state) == self.n_items

    def finalize_solution(self, state):
        return list(state)

    def pheromone_key(self, state, move):
        item_idx = len(state)
        return (item_idx, int(move))

    def all_components(self):
        return [(i, 0) for i in range(self.n_items)] + [(i, 1) for i in range(self.n_items)]

    def solution_components(self, solution):
        return [(i, int(bit)) for i, bit in enumerate(solution)]

    def heuristic_value(self, state, move):
        item_idx = len(state)

        if item_idx >= self.n_items:
            return 1.0

        # Skipping gets a neutral heuristic
        if int(move) == 0:
            return 1.0

        # Taking uses value/weight ratio
        weight = self.weights[item_idx]
        value = self.values[item_idx]

        if weight <= 0:
            return value + 1e-10

        return float(value / weight)

    def solution_to_array(self, solution):
        return np.array(solution, dtype=int)

    # ----------------------------
    # Local-search neighborhood
    # ----------------------------
    def neighbors(self, solution):
        solution = list(solution)

        for i in range(self.n_items):
            neighbor = solution[:]
            neighbor[i] = 1 - neighbor[i]

            if self._solution_weight(neighbor) <= self.capacity:
                yield neighbor

    # ----------------------------
    # Helpers
    # ----------------------------
    def _state_weight(self, state):
        if not state:
            return 0.0
        arr = np.asarray(state, dtype=int)
        return float(np.dot(arr, self.weights[:len(arr)]))

    def _solution_weight(self, solution):
        arr = np.asarray(solution, dtype=int)
        return float(np.dot(arr, self.weights))


def create_knapsack_problem(
    weights=None,
    values=None,
    capacity=15,
    seed=0,
    n_items=10,
    name="knapsack"
):
    """
    Factory for either:
    - user-provided weights/values
    - or a random 0/1 knapsack instance
    """
    if weights is not None and values is not None:
        return KnapsackProblem(weights=weights, values=values, capacity=capacity, name=name)

    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 11, size=n_items)
    values = rng.integers(5, 31, size=n_items)

    return KnapsackProblem(
        weights=weights,
        values=values,
        capacity=capacity,
        name=f"{name}-{n_items}"
    )


# Optional fallback instance
knapsack_problem = create_knapsack_problem()