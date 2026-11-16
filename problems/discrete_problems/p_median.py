"""
The P-Median problem: given a set of demand points and potential facility locations,
select P facilities to minimize the total distance from demand points to their nearest facility.
"""

import numpy as np
from core.problem import OptimizationProblem

class PMedianProblem(OptimizationProblem):
    """
    P-Median Facility Location Problem
    
    Given N locations and P facilities to place, minimize total assignment cost.

    """
    
    def __init__(self, distance_matrix, num_facilities, demand=None):
        """
        Initialize the P-Median problem.

        """
        self.distance_matrix = np.array(distance_matrix)
        self.num_locations = len(distance_matrix)
        self.num_facilities = num_facilities
        
        if demand is None:
            self.demand = np.ones(self.num_locations)
        else:
            self.demand = np.array(demand)
        
        super().__init__(
            name=f"P-Median (P={num_facilities}, N={self.num_locations})",
            objective_function=self.evaluate,
            lower_bounds=[0] * self.num_locations,
            upper_bounds=[1] * self.num_locations,
            is_minimization=True
        )
    
    def evaluate(self, solution):
        """
        Evaluate a P-Median solution.
        
        """
        solution = np.array(solution, dtype=int)
        num_selected = np.sum(solution)
        
        # Penalty for wrong number of facilities
        if num_selected != self.num_facilities:
            penalty = 10000 * abs(num_selected - self.num_facilities)
            return penalty + self._calculate_cost(solution)
        
        return self._calculate_cost(solution)
    
    def _calculate_cost(self, solution):
        """
        Calculate the total assignment cost.
        Each location is assigned to its nearest selected facility.
        """
        facility_indices = np.where(solution == 1)[0]
        
        if len(facility_indices) == 0:
            return float('inf')
        
        total_cost = 0.0
        for i in range(self.num_locations):
            min_distance = np.min(self.distance_matrix[i, facility_indices])
            total_cost += self.demand[i] * min_distance
        
        return total_cost
    
    def get_facility_locations(self, solution):

        return np.where(np.array(solution, dtype=int) == 1)[0]
    
    def get_assignments(self, solution):
        """
        Get assignment of each location to nearest facility.
        
        """
        facility_indices = self.get_facility_locations(solution)
        if len(facility_indices) == 0:
            return None
        
        assignments = np.zeros(self.num_locations, dtype=int)
        for i in range(self.num_locations):
            distances = self.distance_matrix[i, facility_indices]
            nearest_idx = np.argmin(distances)
            assignments[i] = facility_indices[nearest_idx]
        
        return assignments


def create_p_median_problem(num_locations=30, num_facilities=5, seed=42):

    np.random.seed(seed)
    
    # Generate random 2D coordinates
    coordinates = np.random.rand(num_locations, 2) * 100
    
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    
    demand = np.random.randint(1, 20, size=num_locations)
    
    return PMedianProblem(distance_matrix, num_facilities, demand)


def create_small_p_median():
    """Small test instance"""
    # 10 locations, simple grid
    coordinates = np.array([
        [0, 0], [0, 10], [0, 20],
        [10, 0], [10, 10], [10, 20],
        [20, 0], [20, 10], [20, 20],
        [15, 15]
    ])
    
    num_locations = len(coordinates)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    
    return PMedianProblem(distance_matrix, num_facilities=3)


# Instance for config
p_median_problem = create_p_median_problem()