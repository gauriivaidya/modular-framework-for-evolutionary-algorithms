import numpy as np
from core.optimizer import Optimizer
import math


class Bee:
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.fitness = float('inf')
        self.trial_count = 0


class ArtificialBeeColony(Optimizer):
    def __init__(self, problem, population_size=50, num_generations=100, limit=100):
        super().__init__(problem, population_size, num_generations)
        self.limit = limit
        self.num_employed = self.population_size // 2
        self.num_onlooker = self.population_size - self.num_employed

    def optimize(self):
        self.population = [Bee(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds) for _ in
                           range(self.num_employed)]

        for bee in self.population:
            bee.fitness, _, _ = self.problem.evaluate(bee.position)
            if bee.fitness < self.gbest_fitness:
                self.gbest_fitness = bee.fitness
                self.gbest_position = bee.position.copy()

        for gen in range(self.num_generations):
            self._employed_bee_phase()
            self._onlooker_bee_phase()
            self._scout_bee_phase()

            current_best_bee = min(self.population, key=lambda b: b.fitness)
            if current_best_bee.fitness < self.gbest_fitness:
                self.gbest_fitness = current_best_bee.fitness
                self.gbest_position = current_best_bee.position.copy()

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness

    def _employed_bee_phase(self):
        for i in range(self.num_employed):
            bee = self.population[i]

            partner_idx = np.random.randint(0, self.num_employed)
            while partner_idx == i:
                partner_idx = np.random.randint(0, self.num_employed)
            partner_bee = self.population[partner_idx]

            dim_to_change = np.random.randint(0, self.problem.dim)
            phi = np.random.uniform(-1, 1)

            new_position = bee.position.copy()
            new_position[dim_to_change] += phi * (bee.position[dim_to_change] - partner_bee.position[dim_to_change])
            new_position = np.clip(new_position, self.problem.lower_bounds, self.problem.upper_bounds)

            new_fitness, _, _ = self.problem.evaluate(new_position)

            if new_fitness < bee.fitness:
                bee.position = new_position
                bee.fitness = new_fitness
                bee.trial_count = 0
            else:
                bee.trial_count += 1

    def _onlooker_bee_phase(self):
        fitness_values = np.array([1 / (b.fitness + 1e-100) for b in self.population])
        probabilities = fitness_values / np.sum(fitness_values)

        for _ in range(self.num_onlooker):
            chosen_idx = np.random.choice(self.num_employed, p=probabilities)
            bee = self.population[chosen_idx]

            partner_idx = np.random.randint(0, self.num_employed)
            while partner_idx == chosen_idx:
                partner_idx = np.random.randint(0, self.num_employed)
            partner_bee = self.population[partner_idx]

            dim_to_change = np.random.randint(0, self.problem.dim)
            phi = np.random.uniform(-1, 1)

            new_position = bee.position.copy()
            new_position[dim_to_change] += phi * (bee.position[dim_to_change] - partner_bee.position[dim_to_change])
            new_position = np.clip(new_position, self.problem.lower_bounds, self.problem.upper_bounds)

            new_fitness, _, _ = self.problem.evaluate(new_position)

            if new_fitness < bee.fitness:
                bee.position = new_position
                bee.fitness = new_fitness
                bee.trial_count = 0
            else:
                bee.trial_count += 1

    def _scout_bee_phase(self):
        for i in range(self.num_employed):
            if self.population[i].trial_count >= self.limit:
                self.population[i] = Bee(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds)
                self.population[i].fitness, _, _ = self.problem.evaluate(self.population[i].position)
