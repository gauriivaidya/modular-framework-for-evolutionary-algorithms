"""
The 0/1 Knapsack problem: given a set of items with weights and values,
select items to maximize value without exceeding capacity.
"""

import numpy as np
from core.problem import OptimizationProblem

class KnapsackProblem(OptimizationProblem):
    
    def __init__(self, weights, values, capacity):

        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.num_items = len(weights)
        
        super().__init__(
            name="Knapsack",
            objective_function=self.evaluate,
            lower_bounds=[0] * self.num_items,
            upper_bounds=[1] * self.num_items,
            is_minimization=True
        )
    def evaluate(self, solution):
        """
        Evaluate a knapsack solution (binary representation).
        
        Args:
            solution: Binary array where 1 means item is selected
            
        Returns:
            Negative total value (for minimization), or -inf if over capacity
        """
        solution = np.array(solution, dtype=int)
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)
        
        # Penalty for exceeding capacity
        if total_weight > self.capacity:

            excess = total_weight - self.capacity
            return -(total_value - 1000 * excess)
        
        # (GA minimizes by default)
        return -total_value
    
    def get_total_weight(self, solution):
        return np.sum(np.array(solution, dtype=int) * self.weights)
    
    def get_total_value(self, solution):
        return np.sum(np.array(solution, dtype=int) * self.values)
    
    def is_feasible(self, solution):
        return self.get_total_weight(solution) <= self.capacity


# Factory function for creating standard instances
def create_knapsack_problem(num_items=20, capacity=None, seed=42):

    np.random.seed(seed)
    weights = np.random.randint(1, 50, size=num_items)
    values = np.random.randint(1, 100, size=num_items)
    
    if capacity is None:
        capacity = int(np.sum(weights) * 0.5)
    
    return KnapsackProblem(weights, values, capacity)


# Standard benchmark instance
def create_standard_knapsack():
    """Create a standard 10-item knapsack problem for testing."""
    weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
    values = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    capacity = 165
    return KnapsackProblem(weights, values, capacity)


# Default instance for config
knapsack_problem = create_knapsack_problem()
