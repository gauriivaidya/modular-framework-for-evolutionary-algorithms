"""
The Graph Coloring problem: assign colors to vertices such that no two adjacent
vertices have the same color, minimizing the number of colors used.

"""

import numpy as np
from core.problem import OptimizationProblem

class GraphColoringProblem(OptimizationProblem):

    
    def __init__(self, adjacency_matrix, max_colors=None):
        #Initialize the Graph Coloring problem.
        
        self.adjacency_matrix = np.array(adjacency_matrix)
        self.num_vertices = len(adjacency_matrix)
        
        if max_colors is None:
            max_colors = self.num_vertices
        self.max_colors = max_colors
        
        super().__init__(
            name=f"Graph Coloring ({self.num_vertices} vertices)",
            objective_function=self.evaluate,
            lower_bounds=[0] * self.num_vertices,
            upper_bounds=[max_colors - 1] * self.num_vertices,
            is_minimization=True
        )
            
    def evaluate(self, solution):
        #Evaluate a graph coloring solution.
        
        solution = np.array(solution, dtype=int)
        
        # Count conflicts (adjacent vertices with same color)
        conflicts = 0
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if self.adjacency_matrix[i, j] == 1:
                    if solution[i] == solution[j]:
                        conflicts += 1
        
        # Count number of distinct colors used
        num_colors = len(np.unique(solution))
        
        # Minimize colors + heavy penalty for conflicts
        return num_colors + 1000 * conflicts
    
    def get_num_colors(self, solution):
        return len(np.unique(solution))
    
    def get_conflicts(self, solution):
        solution = np.array(solution, dtype=int)
        conflicts = 0
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if self.adjacency_matrix[i, j] == 1:
                    if solution[i] == solution[j]:
                        conflicts += 1
        return conflicts
    
    def is_valid_coloring(self, solution):
        """Check if coloring is valid (no conflicts)."""
        return self.get_conflicts(solution) == 0


def create_random_graph(num_vertices=20, edge_probability=0.3, seed=42):
    #Create a random graph coloring problem.

    np.random.seed(seed)
    adjacency_matrix = np.zeros((num_vertices, num_vertices))
    
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if np.random.rand() < edge_probability:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    
    return GraphColoringProblem(adjacency_matrix)


def create_petersen_graph():
    """Create the famous Petersen graph (3-colorable)."""
    # Petersen graph has 10 vertices
    adjacency_matrix = np.zeros((10, 10))
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Connecting edges
    ]
    
    for i, j in edges:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    
    return GraphColoringProblem(adjacency_matrix, max_colors=10)


def create_graph_coloring_problem(num_vertices=15, edge_probability=0.4, seed=42):
    return create_random_graph(num_vertices, edge_probability, seed)


graph_coloring_problem = create_graph_coloring_problem()