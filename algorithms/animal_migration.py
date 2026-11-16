import numpy as np
from core.optimizer import Optimizer


class Animal:
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.fitness = float('inf')


class AnimalMigrationOptimization(Optimizer):
    def __init__(self, problem, population_size=50, num_generations=100):
        super().__init__(problem, population_size, num_generations)

    def optimize(self):
        self.population = [Animal(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds) for _ in
                           range(self.population_size)]

        for gen in range(self.num_generations):
            for animal in self.population:
                animal.fitness, _, _ = self.problem.evaluate(animal.position)
                if animal.fitness < self.gbest_fitness:
                    self.gbest_fitness = animal.fitness
                    self.gbest_position = animal.position.copy()

            self._migration_phase()
            self._updating_phase()

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness

    def _migration_phase(self):
        current_positions = [p.position for p in self.population]
        for i, animal in enumerate(self.population):
            # Ring topology neighborhood of 5
            indices = [(i - 2 + self.population_size) % self.population_size,
                       (i - 1 + self.population_size) % self.population_size,
                       i,
                       (i + 1) % self.population_size,
                       (i + 2) % self.population_size]

            neighbor_positions = [current_positions[j] for j in indices]
            random_neighbor_pos = neighbor_positions[np.random.randint(0, 5)]

            dg = np.random.normal(0, 1)
            animal.position += dg * (random_neighbor_pos - animal.position)
            animal.position = np.clip(animal.position, self.problem.lower_bounds, self.problem.upper_bounds)

    def _updating_phase(self):
        if np.random.rand() > (1 / self.population_size):
            return

        current_positions = [p.position.copy() for p in self.population]
        worst_animal = max(self.population, key=lambda a: a.fitness)

        t1, t2 = np.random.choice(self.population_size, 2, replace=False)

        new_pos = current_positions[t1] + \
                  np.random.rand() * (self.gbest_position - worst_animal.position) + \
                  np.random.rand() * (current_positions[t2] - worst_animal.position)

        new_pos = np.clip(new_pos, self.problem.lower_bounds, self.problem.upper_bounds)
        new_fitness, _, _ = self.problem.evaluate(new_pos)

        if new_fitness < worst_animal.fitness:
            worst_animal.position = new_pos

