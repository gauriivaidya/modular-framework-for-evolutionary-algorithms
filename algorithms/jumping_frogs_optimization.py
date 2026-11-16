import numpy as np
from core.swarm_optimizer import SwarmOptimizer, SwarmParticle


class Frog(SwarmParticle):
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        super().__init__(num_dimensions, lower_bounds, upper_bounds)


class JumpingFrogsOptimization(SwarmOptimizer):
    def __init__(self, problem, population_size=50, num_generations=100,
                 c1=0.1, c2=0.2, c3=0.5, k_neighbors=15):
        super().__init__(problem, population_size, num_generations, particle_class=Frog)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = 1.0 - c1 - c2 - c3
        self.k_neighbors = k_neighbors

    def _update_positions(self, current_generation):
        for frog in self.population:
            self._perform_jump(frog)

    def _perform_jump(self, frog):
        neighbor_indices = np.random.choice(self.population_size, self.k_neighbors, replace=False)
        neighbors = [self.population[i] for i in neighbor_indices]
        best_neighbor = min(neighbors, key=lambda n: n.pbest_fitness)

        r = np.random.rand()
        temp_c = 0
        if r <= self.c1:
            temp_c = self.c1
            frog.position *= self.c1
        elif r <= self.c1 + self.c2:
            temp_c = self.c2
            frog.position = self.c2 * frog.pbest_position
        elif r <= self.c1 + self.c2 + self.c3:
            temp_c = self.c3
            frog.position = self.c3 * best_neighbor.position
        else:
            temp_c = self.c4
            frog.position = self.c4 * self.gbest_position

        frog.position = np.clip(frog.position, self.problem.lower_bounds, self.problem.upper_bounds)

        g_p = 1 / (temp_c + 1) if temp_c != -1 else 1
        a = np.random.geometric(g_p)
        if a * np.random.rand() * r > 1:
            self._perform_jump(frog)

