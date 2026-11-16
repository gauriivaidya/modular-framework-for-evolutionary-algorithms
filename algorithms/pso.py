import numpy as np
from core.swarm_optimizer import SwarmOptimizer, SwarmParticle
from core.problem import OptimizationProblem


class Particle(SwarmParticle):
    """A particle for the PSO algorithm, including velocity."""

    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        super().__init__(num_dimensions, lower_bounds, upper_bounds)
        self.velocity = np.zeros(num_dimensions)


class PSO(SwarmOptimizer):
    def __init__(self, problem: OptimizationProblem, population_size: int, num_generations: int, w=0.9, c1=0.2, c2=0.8):
        """
        Initializes the PSO optimizer.

        Args:
            problem (OptimizationProblem): The problem to be solved.
            population_size (int): The size of the swarm.
            num_generations (int): The number of generations to run.
            w (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
        """
        super().__init__(problem, population_size, num_generations)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _initialize_population(self):
        """Creates the initial population of PSO particles."""
        return [Particle(self.problem.num_dimensions, self.problem.lower_bounds, self.problem.upper_bounds)
                for _ in range(self.population_size)]

    def _update_positions(self, current_generation):
        """Updates particle velocities and positions according to the PSO rules."""
        for particle in self.population:
            r1 = np.random.rand(self.problem.num_dimensions)
            r2 = np.random.rand(self.problem.num_dimensions)

            cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
            social = self.c2 * r2 * (self.gbest_position - particle.position)

            particle.velocity = self.w * particle.velocity + cognitive + social
            particle.position += particle.velocity

            particle.position = np.clip(particle.position, self.problem.lower_bounds, self.problem.upper_bounds)
