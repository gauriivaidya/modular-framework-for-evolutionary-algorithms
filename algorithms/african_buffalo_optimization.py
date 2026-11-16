import numpy as np
from core.swarm_optimizer import SwarmOptimizer, SwarmParticle


class Buffalo(SwarmParticle):
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        super().__init__(num_dimensions, lower_bounds, upper_bounds)
        self.velocity = np.zeros(num_dimensions)


class AfricanBuffaloOptimization(SwarmOptimizer):
    def __init__(self, problem, population_size=50, num_generations=100,
                 lp1=0.2, lp2=0.8):
        super().__init__(problem, population_size, num_generations, particle_class=Buffalo)
        self.lp1 = lp1
        self.lp2 = lp2
        self.l_factor = 1 / np.random.uniform(0, 1)

    def _update_positions(self, current_generation):
        for buffalo in self.population:
            r1 = np.random.rand()
            r2 = np.random.rand()

            new_velocity = buffalo.velocity + \
                           (self.lp1 * r1 * (self.gbest_position - buffalo.position)) + \
                           (self.lp2 * r2 * (buffalo.pbest_position - buffalo.position))
            buffalo.velocity = new_velocity

            buffalo.position = self.l_factor * (buffalo.position + buffalo.velocity)
            buffalo.position = np.clip(buffalo.position, self.problem.lower_bounds, self.problem.upper_bounds)

