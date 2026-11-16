import numpy as np
from core.optimizer import Optimizer
import math


class Fish:
    def __init__(self, num_dimensions, lower_bounds, upper_bounds, fish_type):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.fitness = float('inf')
        self.type = fish_type


class SailfishOptimizer(Optimizer):
    def __init__(self, problem, population_size=100, num_generations=100,
                 s_ratio=0.2, A=4, e=1e-5):
        super().__init__(problem, population_size, num_generations)
        self.num_sailfish = int(population_size * s_ratio)
        self.num_sardines = population_size - self.num_sailfish
        self.A = A
        self.e = e

    def optimize(self):
        sailfish_pop = [Fish(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds, 'sailfish') for _
                        in range(self.num_sailfish)]
        sardine_pop = [Fish(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds, 'sardine') for _ in
                       range(self.num_sardines)]

        for gen in range(self.num_generations):
            for sf in sailfish_pop: sf.fitness, _, _ = self.problem.evaluate(sf.position)
            for sa in sardine_pop: sa.fitness, _, _ = self.problem.evaluate(sa.position)

            elite_sailfish = min(sailfish_pop, key=lambda f: f.fitness)
            injured_sardine = min(sardine_pop, key=lambda f: f.fitness)

            if elite_sailfish.fitness < self.gbest_fitness:
                self.gbest_fitness = elite_sailfish.fitness
                self.gbest_position = elite_sailfish.position.copy()

            pd = 1 - (len(sailfish_pop) / (len(sailfish_pop) + len(sardine_pop)))

            # Update Sailfish positions
            for sf in sailfish_pop:
                lambda_i = 2 * np.random.rand() * pd - pd
                sf.position = elite_sailfish.position - lambda_i * (
                            np.random.rand() * ((elite_sailfish.position + injured_sardine.position) / 2) - sf.position)
                sf.position = np.clip(sf.position, self.problem.lower_bounds, self.problem.upper_bounds)

            # Update Sardine positions
            ap = self.A * (1 - (2 * (gen + 1) * self.e))
            for sa in sardine_pop:
                sa.position = np.random.rand() * (elite_sailfish.position - sa.position + ap)
                sa.position = np.clip(sa.position, self.problem.lower_bounds, self.problem.upper_bounds)

            # Hunting and updating populations
            for sf in sailfish_pop:
                for sa in sardine_pop[:]:
                    if np.linalg.norm(sf.position - sa.position) < 1e-5:  # Simplified capture
                        sf.position = sa.position
                        sardine_pop.remove(sa)
                        break

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness
