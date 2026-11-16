import numpy as np
from core.optimizer import Optimizer


class Bird:
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('inf')
        self.role = "scrounger"


class BirdSwarmAlgorithm(Optimizer):
    def __init__(self, problem, population_size=50, num_generations=100,
                 c1=1.5, c2=1.5, a1=1.0, a2=1.0, fq=3, p_producer=0.9, fl=0.7):
        super().__init__(problem, population_size, num_generations)
        self.c1 = c1
        self.c2 = c2
        self.a1 = a1
        self.a2 = a2
        self.fq = fq
        self.p_producer = p_producer
        self.fl = fl

    def optimize(self):
        self.population = [Bird(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds) for _ in
                           range(self.population_size)]

        for gen in range(self.num_generations):
            for bird in self.population:
                fitness, _, _ = self.problem.evaluate(bird.position)
                if fitness < bird.pbest_fitness:
                    bird.pbest_fitness = fitness
                    bird.pbest_position = bird.position.copy()
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = bird.position.copy()

            if (gen + 1) % self.fq == 0:
                self._update_roles()
                self._flight_behavior()
            else:
                self._foraging_behavior()

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness

    def _update_roles(self):
        sorted_birds = sorted(self.population, key=lambda b: b.pbest_fitness)
        best_bird = sorted_birds[0]
        worst_bird = sorted_birds[-1]

        for bird in self.population:
            if bird is best_bird:
                bird.role = "producer"
            elif bird is worst_bird:
                bird.role = "scrounger"
            else:
                bird.role = "producer" if np.random.rand() < self.p_producer else "scrounger"

    def _flight_behavior(self):
        mean_position = np.mean([b.position for b in self.population], axis=0)
        pbest_fitness_sum = np.sum([b.pbest_fitness for b in self.population if np.isfinite(b.pbest_fitness)]) + 1e-100

        for bird in self.population:
            if bird.role == "producer":
                r1, r2 = np.random.rand(2)
                update = self.c1 * r1 * (bird.pbest_position - bird.position) + \
                         self.c2 * r2 * (self.gbest_position - bird.position)
                bird.position += update
            else:
                k = np.random.randint(0, self.population_size)
                other_bird = self.population[k]
                if other_bird is bird: continue

                A1 = self.a1 * np.exp(-bird.pbest_fitness * self.population_size / pbest_fitness_sum)
                denominator = abs(other_bird.pbest_fitness - bird.pbest_fitness) + 1e-100
                A2 = self.a2 * np.exp((other_bird.pbest_fitness - bird.pbest_fitness) / denominator * (
                            self.population_size * bird.pbest_fitness / pbest_fitness_sum))

                update = A1 * np.random.rand() * (mean_position - bird.position) + \
                         A2 * np.random.uniform(-1, 1) * (other_bird.pbest_position - bird.position)
                bird.position += update

            bird.position = np.clip(bird.position, self.problem.lower_bounds, self.problem.upper_bounds)

    def _foraging_behavior(self):
        for bird in self.population:
            if bird.role == "producer":
                update = bird.position * np.random.normal(0, 1)
                bird.position += update
            else:
                producers = [b for b in self.population if b.role == "producer"]
                if producers:
                    producer_to_follow = producers[np.random.randint(0, len(producers))]
                    update = (producer_to_follow.position - bird.position) * self.fl * np.random.rand()
                    bird.position += update

            bird.position = np.clip(bird.position, self.problem.lower_bounds, self.problem.upper_bounds)

