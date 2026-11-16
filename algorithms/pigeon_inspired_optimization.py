import numpy as np
from core.optimizer import Optimizer


class Pigeon:
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.velocity = np.zeros(num_dimensions)
        self.fitness = float('inf')


class PigeonInspiredOptimization(Optimizer):
    def __init__(self, problem, population_size=50, num_generations=100,
                 R=0.2, nc1_max=150, nc2_max=200):
        super().__init__(problem, population_size, num_generations)
        self.R = R
        self.nc1_max = nc1_max
        self.nc2_max = nc2_max

    def optimize(self):
        self.population = [Pigeon(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds) for _ in
                           range(self.population_size)]

        for gen in range(self.num_generations):
            for pigeon in self.population:
                pigeon.fitness, _, _ = self.problem.evaluate(pigeon.position)

            self.population.sort(key=lambda p: p.fitness)
            if self.population[0].fitness < self.gbest_fitness:
                self.gbest_fitness = self.population[0].fitness
                self.gbest_position = self.population[0].position.copy()

            if gen < self.nc1_max:
                self._map_and_compass_operator(gen)
            else:
                self._landmark_operator()

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness

    def _map_and_compass_operator(self, gen):
        for pigeon in self.population:
            r = np.random.rand()
            new_velocity = pigeon.velocity * np.exp(-self.R * (gen + 1)) + r * (self.gbest_position - pigeon.position)
            pigeon.velocity = new_velocity
            pigeon.position += pigeon.velocity
            pigeon.position = np.clip(pigeon.position, self.problem.lower_bounds, self.problem.upper_bounds)

    def _landmark_operator(self):
        # Halve the population
        num_pigeons = len(self.population) // 2
        self.population = self.population[:num_pigeons]
        if not self.population: return

        # Calculate center
        weighted_sum = np.sum([p.position / (p.fitness + 1e-100) for p in self.population], axis=0)
        sum_of_inv_fitness = np.sum([1 / (p.fitness + 1e-100) for p in self.population])
        center = weighted_sum / sum_of_inv_fitness

        for pigeon in self.population:
            r = np.random.rand()
            pigeon.position += r * (center - pigeon.position)
            pigeon.position = np.clip(pigeon.position, self.problem.lower_bounds, self.problem.upper_bounds)
