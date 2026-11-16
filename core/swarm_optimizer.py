import numpy as np
import logging
from .optimizer import Optimizer
from .problem import OptimizationProblem

logger = logging.getLogger(__name__)


class SwarmParticle:
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('inf')


class SwarmOptimizer(Optimizer):
    def __init__(self, problem: OptimizationProblem, population_size: int, num_generations: int):
        super().__init__(problem, population_size, num_generations)
        self.population = self._initialize_population()
        self.history = []

    def _initialize_population(self):
        raise NotImplementedError("Subclasses must implement _initialize_population.")

    def _update_positions(self, current_generation):
        raise NotImplementedError("Subclasses must implement _update_positions.")

    def optimize(self):
        for gen in range(self.num_generations):
            for particle in self.population:
                fitness, _, _ = self.problem.evaluate(particle.position)
                if self.problem.is_minimization:
                    if fitness < particle.pbest_fitness:
                        particle.pbest_fitness = fitness
                        particle.pbest_position = particle.position.copy()
                else:
                    if fitness > particle.pbest_fitness:
                        particle.pbest_fitness = fitness
                        particle.pbest_position = particle.position.copy()

                self._update_global_best(particle.position, fitness, current_generation=gen + 1)

            self._update_positions(gen)

            self.history.append(
                {'generation': gen + 1, 'fitness': self.gbest_fitness, 'solution': self.gbest_position.copy()})
            logger.info(f"Generation {gen + 1}/{self.num_generations} | Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness
