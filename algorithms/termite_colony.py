import numpy as np
from core.optimizer import Optimizer


class Termite:
    """Represents a termite in the Termite Colony Optimization."""

    def __init__(self, num_dimensions, lower_bounds, upper_bounds, t_max):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.fitness = float('inf')
        self.pheromone = 0.0
        self.trail = t_max.copy()


class TermiteColonyOptimization(Optimizer):
    """
    The Termite Colony Optimization (TCO) algorithm.
    """

    def __init__(self, problem, population_size=50, num_generations=100, rho=0.8):
        super().__init__(problem, population_size, num_generations)
        self.rho = rho
        ub = self.problem.upper_bounds
        lb = self.problem.lower_bounds
        self.t_max = 0.5 * (ub - lb)
        self.t_min = 0.125 * (ub - lb)
        self.neighbor_range = 1 / 6 * (ub - lb)

    def optimize(self):
        self.population = [Termite(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds, self.t_max)
                           for _ in range(self.population_size)]

        for gen in range(self.num_generations):
            for termite in self.population:
                termite.fitness, _, _ = self.problem.evaluate(termite.position)
                termite.pheromone = (1 - self.rho) * termite.pheromone + (1 / (termite.fitness + 1))
                if termite.fitness < self.gbest_fitness:
                    self.gbest_fitness = termite.fitness
                    self.gbest_position = termite.position.copy()

            for termite in self.population:
                neighbors = [other for other in self.population if
                             np.linalg.norm(termite.position - other.position) < np.linalg.norm(self.neighbor_range)]

                if neighbors:
                    best_neighbor = min(neighbors, key=lambda n: n.fitness)

                    if termite.pheromone < best_neighbor.pheromone:
                        wb = np.random.uniform(1, 2)
                        rb = np.random.rand()
                        termite.position += wb * rb * (best_neighbor.position - termite.position)
                    else:
                        rw = np.random.uniform(-np.abs(termite.trail), termite.trail)
                        termite.position += np.random.uniform(rw,
                                                              termite.position)  # Original logic was slightly ambiguous
                else:
                    rw = np.random.uniform(-np.abs(termite.trail), termite.trail)
                    termite.position += np.random.uniform(rw, termite.position)

                termite.trail -= (self.t_max - self.t_min) / self.num_generations
                termite.position = np.clip(termite.position, self.problem.lower_bounds, self.problem.upper_bounds)

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness
