import numpy as np
from core.problem import OptimizationProblem


class GeneralKnapsackProblem(OptimizationProblem):

    def __init__(self, weights, values, capacity, max_counts):

        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.max_counts = np.array(max_counts)
        self.num_items = len(weights)

        super().__init__(
            name="GeneralKnapsack",
            objective_function=self.evaluate,
            lower_bounds=[0] * self.num_items,
            upper_bounds=self.max_counts.tolist(),   # 🔥 important
            is_minimization=True
        )

    def evaluate(self, solution):
        """
        Integer knapsack evaluation.
        solution[i] = number of copies of item i
        """

        solution = np.array(solution, dtype=int)

        # Ensure bounds (safety)
        solution = np.clip(solution, 0, self.max_counts)

        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)

        # Penalty for exceeding capacity
        if total_weight > self.capacity:
            excess = total_weight - self.capacity
            penalty = excess * np.mean(self.values)
            return -(total_value - penalty)

        return -total_value

    def get_total_weight(self, solution):
        return np.sum(np.array(solution, dtype=int) * self.weights)

    def get_total_value(self, solution):
        return np.sum(np.array(solution, dtype=int) * self.values)

    def is_feasible(self, solution):
        return self.get_total_weight(solution) <= self.capacity

def create_general_knapsack_problem(
    num_items=20,
    capacity=None,
    capacity_ratio=0.5,
    max_count=5,
    seed=42
):
    np.random.seed(seed)

    weights = np.random.randint(1, 50, size=num_items)
    values = np.random.randint(1, 100, size=num_items)

    if capacity is None:
        capacity = int(np.sum(weights) * capacity_ratio)

    max_counts = np.random.randint(1, max_count + 1, size=num_items)

    return GeneralKnapsackProblem(weights, values, capacity, max_counts)