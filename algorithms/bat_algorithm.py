import numpy as np
import logging
from core.swarm_optimizer import SwarmOptimizer, SwarmParticle
from core.problem import OptimizationProblem

logger = logging.getLogger(__name__)


class Bat(SwarmParticle):
    """A bat for the Bat Algorithm, including velocity, frequency, rate, and loudness."""

    def __init__(self, num_dimensions, lower_bounds, upper_bounds, fmin, fmax):
        super().__init__(num_dimensions, lower_bounds, upper_bounds)
        self.velocity = np.zeros(num_dimensions)
        self.frequency = fmin + (fmax - fmin) * np.random.rand()
        self.rate = 1.0
        self.loudness = 1.0


class BatAlgorithm(SwarmOptimizer):

    def __init__(self, problem: OptimizationProblem, population_size: int, num_generations: int,
                 alpha=0.97, gamma=0.1, fmin=0.0, fmax=2.0):
        """
        Initializes the Bat Algorithm optimizer.

        Args:
            problem (OptimizationProblem): The problem to be solved.
            population_size (int): The size of the bat population.
            num_generations (int): The number of generations to run.
            alpha (float): Rate of loudness decrease.
            gamma (float): Rate of pulse rate increase.
            fmin (float): Minimum frequency.
            fmax (float): Maximum frequency.
        """
        super().__init__(problem, population_size, num_generations)
        self.alpha = alpha
        self.gamma = gamma
        self.fmin = fmin
        self.fmax = fmax

    def _initialize_population(self):
        """Creates the initial population of bats."""
        return [
            Bat(self.problem.num_dimensions, self.problem.lower_bounds, self.problem.upper_bounds, self.fmin, self.fmax)
            for _ in range(self.population_size)]

    def _update_positions(self, current_generation):
        """Updates the positions of the bats based on BA logic."""
        avg_loudness = np.mean([bat.loudness for bat in self.population])

        for bat in self.population:
            bat.velocity += (bat.position - self.gbest_position) * bat.frequency
            new_position = bat.position + bat.velocity

            if np.random.rand() > bat.rate:
                new_position = self.gbest_position + np.random.uniform(-1, 1,
                                                                       self.problem.num_dimensions) * avg_loudness

            new_position = np.clip(new_position, self.problem.lower_bounds, self.problem.upper_bounds)

            new_fitness, _, _ = self.problem.evaluate(new_position)

            if np.random.rand() < bat.loudness and new_fitness < self.gbest_fitness:
                bat.position = new_position
                self._update_global_best(new_position, new_fitness)

                bat.loudness *= self.alpha
                bat.rate = 1 - np.exp(-self.gamma * current_generation)
