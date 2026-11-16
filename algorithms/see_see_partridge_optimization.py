import numpy as np
from core.optimizer import Optimizer


class Partridge(object):
    """Represents a partridge in the SSPOA algorithm."""

    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.velocity = np.zeros(num_dimensions)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('inf')
        self.priority = 0


class SeeSeePartridgeOptimization(Optimizer):
    """
    The See See Partridge Optimization Algorithm (SSPOA).
    """

    def __init__(self, problem, population_size=50, num_generations=100,
                 w=0.9, c=0.8):
        super().__init__(problem, population_size, num_generations)
        self.w = w
        self.c = c

    def optimize(self):
        """Runs the SSPOA algorithm."""
        self.population = [Partridge(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds) for _ in
                           range(self.population_size)]

        priority_list = []

        for gen in range(self.num_generations):
            for partridge in self.population:
                fitness, _, _ = self.problem.evaluate(partridge.position)
                if fitness < partridge.pbest_fitness:
                    partridge.pbest_fitness = fitness
                    partridge.pbest_position = partridge.position.copy()
                    partridge.priority += 1
                    priority_list.append((partridge.priority, partridge.position.copy()))

                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = partridge.position.copy()
                    partridge.priority += 1
                    priority_list.append((partridge.priority, partridge.position.copy()))

            priority_list.sort(key=lambda x: x[0])

            for partridge in self.population:
                if len(priority_list) > partridge.priority + 1:
                    target_position = priority_list[partridge.priority + 1][1]

                    r = np.random.rand()
                    new_velocity = (self.w * partridge.velocity) + \
                                   (self.c * r * (target_position - partridge.position))

                    partridge.velocity = new_velocity
                    partridge.position += partridge.velocity
                    partridge.position = np.clip(partridge.position, self.problem.lower_bounds,
                                                 self.problem.upper_bounds)

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness
