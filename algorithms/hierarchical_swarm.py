import numpy as np
from core.optimizer import Optimizer
import math


class Particle:
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.velocity = np.zeros(num_dimensions)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('inf')
        self.subswarm_idx = -1


class HierarchicalSwarmOptimization(Optimizer):
    def __init__(self, problem, population_size=100, num_generations=100,
                 c1=1.3667, c2=1.3667, c3=1.3667, num_subswarms=10):
        super().__init__(problem, population_size, num_generations)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_subswarms = num_subswarms
        phi = c1 + c2 + c3
        self.chi = 2 / abs(2 - phi - math.sqrt(phi ** 2 - 4 * phi))

    def optimize(self):
        self.population = [Particle(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds) for _ in
                           range(self.population_size)]

        subswarm_size = self.population_size // self.num_subswarms
        for i, particle in enumerate(self.population):
            particle.subswarm_idx = i // subswarm_size

        for gen in range(self.num_generations):
            subswarm_bests = self._get_subswarm_bests()

            for particle in self.population:
                fitness, _, _ = self.problem.evaluate(particle.position)
                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = particle.position.copy()
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = particle.position.copy()

            for particle in self.population:
                r2, r3 = np.random.rand(2)
                sbest_position = subswarm_bests[particle.subswarm_idx]

                cognitive_comp = self.c1 * (particle.pbest_position - particle.position)
                subswarm_comp = self.c2 * r2 * (sbest_position - particle.position)
                global_comp = self.c3 * r3 * (self.gbest_position - particle.position)

                particle.velocity = self.chi * (particle.velocity + cognitive_comp + subswarm_comp + global_comp)
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, self.problem.lower_bounds, self.problem.upper_bounds)

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness

    def _get_subswarm_bests(self):
        subswarm_bests = {}
        for i in range(self.num_subswarms):
            subswarm_particles = [p for p in self.population if p.subswarm_idx == i]
            if subswarm_particles:
                best_in_subswarm = min(subswarm_particles, key=lambda p: p.pbest_fitness)
                subswarm_bests[i] = best_in_subswarm.pbest_position
        return subswarm_bests
