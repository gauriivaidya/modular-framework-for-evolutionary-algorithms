import numpy as np
from core.swarm_optimizer import SwarmOptimizer, SwarmParticle
from core.problem import OptimizationProblem


class Whale(SwarmParticle):
    """A particle representing a whale for the WOA."""

    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        super().__init__(num_dimensions, lower_bounds, upper_bounds)


class WhaleOptimization(SwarmOptimizer):
    def __init__(self, problem: OptimizationProblem, population_size: int, num_generations: int, b=1.0, p=0.5):
        super().__init__(problem, population_size, num_generations)
        self.b = b
        self.p = p

    def _initialize_population(self):
        """Creates the initial population of whales."""
        return [Whale(self.problem.num_dimensions, self.problem.lower_bounds, self.problem.upper_bounds)
                for _ in range(self.population_size)]

    def _update_positions(self, current_generation):
        """Updates the positions of the whales based on WOA logic."""
        a = 2 - 2 * (current_generation / self.num_generations)

        for whale in self.population:
            r1 = np.random.rand()
            r2 = np.random.rand()

            A = 2 * a * r1 - a
            C = 2 * r2
            l = np.random.uniform(-1, 1)

            if np.random.rand() < self.p:
                if abs(A) < 1:
                    D = abs(C * self.gbest_position - whale.position)
                    whale.position = self.gbest_position - A * D
                else:
                    random_whale_index = np.random.randint(0, self.population_size)
                    random_whale_pos = self.population[random_whale_index].position
                    D = abs(C * random_whale_pos - whale.position)
                    whale.position = random_whale_pos - A * D
            else:
                D_prime = abs(self.gbest_position - whale.position)
                whale.position = D_prime * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.gbest_position

            # Keep the whale remains within the search space bounds
            whale.position = np.clip(whale.position, self.problem.lower_bounds, self.problem.upper_bounds)
