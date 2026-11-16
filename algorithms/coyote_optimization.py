import numpy as np
from core.optimizer import Optimizer
import math


class Coyote:
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.fitness = float('inf')
        self.age = 0
        self.pack_idx = -1


class CoyoteOptimization(Optimizer):
    def __init__(self, problem, population_size=50, num_generations=100, num_packs=5, coyotes_per_pack=10):
        super().__init__(problem, population_size, num_generations)
        self.num_packs = num_packs
        self.coyotes_per_pack = coyotes_per_pack
        self.population_size = self.num_packs * self.coyotes_per_pack

    def optimize(self):
        self.population = [Coyote(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds) for _ in
                           range(self.population_size)]

        for i, coyote in enumerate(self.population):
            coyote.pack_idx = i // self.coyotes_per_pack
            coyote.fitness, _, _ = self.problem.evaluate(coyote.position)

        self.gbest_position = min(self.population, key=lambda c: c.fitness).position.copy()
        self.gbest_fitness = min(c.fitness for c in self.population)

        for gen in range(self.num_generations):
            packs = [[c for c in self.population if c.pack_idx == i] for i in range(self.num_packs)]

            for i in range(self.num_packs):
                pack = packs[i]
                if not pack: continue

                alpha_coyote = min(pack, key=lambda c: c.fitness)
                tendency = np.mean([c.position for c in pack], axis=0)

                for coyote in pack:
                    r1, r2 = np.random.rand(2)
                    delta1 = alpha_coyote.position - coyote.position
                    delta2 = tendency - coyote.position

                    new_pos = coyote.position + r1 * delta1 + r2 * delta2
                    new_pos = np.clip(new_pos, self.problem.lower_bounds, self.problem.upper_bounds)
                    new_fitness, _, _ = self.problem.evaluate(new_pos)

                    if new_fitness < coyote.fitness:
                        coyote.position = new_pos
                        coyote.fitness = new_fitness

                self._birth_and_death(pack, i)

            for coyote in self.population:
                if coyote.fitness < self.gbest_fitness:
                    self.gbest_fitness = coyote.fitness
                    self.gbest_position = coyote.position.copy()
                coyote.age += 1

            self._coyote_exchange()
            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness

    def _birth_and_death(self, pack, pack_idx):
        if len(pack) < 2: return

        p1, p2 = np.random.choice(pack, 2, replace=False)
        ps = 1 / self.problem.dim
        pa = (1 - ps) / 2

        pup_pos = np.zeros(self.problem.dim)
        for d in range(self.problem.dim):
            r = np.random.rand()
            if r < ps:
                pup_pos[d] = p1.position[d]
            elif r < ps + pa:
                pup_pos[d] = p2.position[d]
            else:
                pup_pos[d] = np.random.uniform(self.problem.lower_bounds[d], self.problem.upper_bounds[d])

        pup_pos = np.clip(pup_pos, self.problem.lower_bounds, self.problem.upper_bounds)
        pup = Coyote(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds)
        pup.position = pup_pos
        pup.fitness, _, _ = self.problem.evaluate(pup.position)
        pup.pack_idx = pack_idx

        worst_coyotes = sorted(pack, key=lambda c: c.fitness, reverse=True)
        if len(worst_coyotes) > 1 and pup.fitness < worst_coyotes[0].fitness:
            if worst_coyotes[0].age > worst_coyotes[1].age:
                self.population.remove(worst_coyotes[0])
                self.population.append(pup)
            else:
                self.population.remove(worst_coyotes[1])
                self.population.append(pup)

    def _coyote_exchange(self):
        prob_leave = 0.005 * (self.coyotes_per_pack ** 2)
        if prob_leave > 1.0: prob_leave = 1.0

        for coyote in self.population:
            if len([c for c in self.population if c.pack_idx == coyote.pack_idx]) > self.coyotes_per_pack:
                if np.random.rand() < prob_leave:
                    coyote.pack_idx = np.random.randint(0, self.num_packs)
