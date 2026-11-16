import numpy as np
from core.swarm_optimizer import SwarmOptimizer, SwarmParticle


class Agent(SwarmParticle):
    """A particle for the Weightless Swarm Optimization algorithm."""

    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        super().__init__(num_dimensions, lower_bounds, upper_bounds)


class WeightlessSwarmOptimization(SwarmOptimizer):

    def __init__(self, problem, population_size=50, num_generations=100,
                 c1=1.7, c2=1.7):
        super().__init__(problem, population_size, num_generations, particle_class=Agent)
        self.c1 = c1
        self.c2 = c2

    def _update_positions(self, current_generation):
        for agent in self.population:
            r1 = np.random.rand()
            r2 = np.random.rand()

            update = (self.c1 * r1) * (self.gbest_position - agent.position) + \
                     (self.c2 * r2) * (agent.pbest_position - agent.position)

            agent.position += update
            agent.position = np.clip(agent.position, self.problem.lower_bounds, self.problem.upper_bounds)
