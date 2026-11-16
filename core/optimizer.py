import numpy as np
import logging
from .problem import OptimizationProblem

logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(self, problem: OptimizationProblem, population_size: int, num_generations: int):
        self.problem = problem
        self.population_size = population_size
        self.num_generations = num_generations
        self.gbest_position = None
        self.gbest_fitness = float('inf') if problem.is_minimization else -float('inf')
        self.convergence_generation = -1

    def optimize(self):
        raise NotImplementedError("Subclasses must implement the optimize method.")

    def _update_global_best(self, position, fitness, current_generation=None):
        is_updated = False
        if self.problem.is_minimization:
            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = position.copy()
                is_updated = True
        else:
            if fitness > self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = position.copy()
                is_updated = True

        if is_updated and current_generation is not None:
            self.convergence_generation = current_generation

        return is_updated
