import numpy as np
from core.optimizer import Optimizer
import math


class Chicken:
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.fitness = float('inf')
        self.role = "chick"
        self.mother_position = None


class ChickenSwarmOptimization(Optimizer):
    def __init__(self, problem, population_size=50, num_generations=100, e=1e-3):
        super().__init__(problem, population_size, num_generations)
        self.e = e
        self.num_roosters = math.floor(0.2 * self.population_size)
        self.num_hens = math.floor(0.6 * self.population_size)
        self.num_mothers = math.floor(0.1 * self.population_size)

    def optimize(self):
        self.population = [Chicken(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds) for _ in
                           range(self.population_size)]

        for gen in range(self.num_generations):
            for chicken in self.population:
                chicken.fitness, _, _ = self.problem.evaluate(chicken.position)

            self.population.sort(key=lambda c: c.fitness)
            roosters, hens, chicks = self._assign_roles()

            if roosters[0].fitness < self.gbest_fitness:
                self.gbest_fitness = roosters[0].fitness
                self.gbest_position = roosters[0].position.copy()

            self._update_roosters(roosters)
            self._update_hens(hens, roosters)
            self._update_chicks(chicks, hens)

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness

    def _assign_roles(self):
        roosters = self.population[:self.num_roosters]
        hens = self.population[self.num_roosters:self.num_roosters + self.num_hens]
        chicks = self.population[self.num_roosters + self.num_hens:]

        for r in roosters: r.role = "rooster"
        for h in hens: h.role = "hen"
        for c in chicks: c.role = "chick"

        if self.num_mothers > 0 and hens:
            mothers = np.random.choice(hens, self.num_mothers, replace=False)
            for chick in chicks:
                chick.mother_position = np.random.choice(mothers).position
        return roosters, hens, chicks

    def _update_roosters(self, roosters):
        for rooster in roosters:
            k = np.random.randint(0, len(roosters))
            other_rooster = roosters[k]

            sigma_sq = 1.0
            if rooster.fitness > other_rooster.fitness:
                sigma_sq = np.exp((other_rooster.fitness - rooster.fitness) / (abs(rooster.fitness) + self.e))

            rooster.position *= (1 + np.random.normal(0, sigma_sq))
            rooster.position = np.clip(rooster.position, self.problem.lower_bounds, self.problem.upper_bounds)

    def _update_hens(self, hens, roosters):
        for hen in hens:
            r1_idx = np.random.randint(0, len(roosters))
            rooster1 = roosters[r1_idx]
            other_chicken_pop = roosters + hens
            other_chicken = other_chicken_pop[np.random.randint(0, len(other_chicken_pop))]

            S1 = np.exp((hen.fitness - rooster1.fitness) / (abs(hen.fitness) + self.e))
            S2 = np.exp(other_chicken.fitness - hen.fitness)

            update1 = S1 * np.random.rand() * (rooster1.position - hen.position)
            update2 = S2 * np.random.rand() * (other_chicken.position - hen.position)
            hen.position += update1 + update2
            hen.position = np.clip(hen.position, self.problem.lower_bounds, self.problem.upper_bounds)

    def _update_chicks(self, chicks, hens):
        for chick in chicks:
            if chick.mother_position is not None:
                FL = np.random.uniform(0, 2)
                chick.position += FL * (chick.mother_position - chick.position)
                chick.position = np.clip(chick.position, self.problem.lower_bounds, self.problem.upper_bounds)

