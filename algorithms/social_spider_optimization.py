import numpy as np
from core.optimizer import Optimizer


class Spider:
    """Represents a spider in the Social Spider Optimization."""

    def __init__(self, num_dimensions, lower_bounds, upper_bounds, gender):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.fitness = float('inf')
        self.weight = 0.0
        self.gender = gender  # 0 for female, 1 for male


class SocialSpiderOptimization(Optimizer):
    """
    The Social Spider Optimization (SSO) algorithm.
    """

    def __init__(self, problem, population_size=50, num_generations=100, pf=0.7):
        super().__init__(problem, population_size, num_generations)
        self.pf = pf
        self.num_females = int(np.floor((0.9 - np.random.rand() * 0.25) * population_size))
        self.num_males = population_size - self.num_females

    def optimize(self):
        self.population = [Spider(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds, 0) for _ in
                           range(self.num_females)] + \
                          [Spider(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds, 1) for _ in
                           range(self.num_males)]

        for gen in range(self.num_generations):
            for spider in self.population:
                spider.fitness, _, _ = self.problem.evaluate(spider.position)

            worst_fitness = max(s.fitness for s in self.population)
            best_fitness = min(s.fitness for s in self.population)

            for spider in self.population:
                spider.weight = (worst_fitness - spider.fitness) / (worst_fitness - best_fitness + 1e-100)

            if best_fitness < self.gbest_fitness:
                self.gbest_fitness = best_fitness
                self.gbest_position = min(self.population, key=lambda s: s.fitness).position.copy()

            self._spider_movements()

            self._mating()

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness

    def _spider_movements(self):
        females = [s for s in self.population if s.gender == 0]
        males = [s for s in self.population if s.gender == 1]

        for spider in self.population:
            if spider.gender == 0:  # Female movement
                others = [s for s in self.population if s.weight > spider.weight]
                if not others: continue
                closest_stronger = min(others, key=lambda s: np.linalg.norm(spider.position - s.position))

                vib_c = closest_stronger.weight * np.exp(
                    -np.linalg.norm(spider.position - closest_stronger.position) ** 2)
                vib_b = self.gbest_position

                alpha, beta, delta = np.random.rand(3)
                attraction = alpha * vib_c * (closest_stronger.position - spider.position)
                attraction_gbest = beta * (self.gbest_position - spider.position)
                random_walk = delta * (np.random.rand(self.problem.dim) - 0.5)

                if np.random.rand() < self.pf:
                    spider.position += attraction + attraction_gbest + random_walk
                else:
                    spider.position -= (attraction + attraction_gbest + random_walk)

            else:
                median_male_weight = np.median([m.weight for m in males]) if males else 0
                if spider.weight > median_male_weight:
                    closest_female = min(females, key=lambda f: np.linalg.norm(
                        spider.position - f.position)) if females else None
                    if closest_female:
                        alpha, delta = np.random.rand(2)
                        vib_f = closest_female.weight * np.exp(
                            -np.linalg.norm(spider.position - closest_female.position) ** 2)
                        spider.position += alpha * vib_f * (closest_female.position - spider.position) + delta * (
                                    np.random.rand(self.problem.dim) - 0.5)
                else:
                    weighted_sum = np.sum([m.position * m.weight for m in males], axis=0)
                    sum_of_weights = np.sum([m.weight for m in males])
                    if sum_of_weights > 0:
                        male_center = weighted_sum / sum_of_weights
                        alpha = np.random.rand()
                        spider.position += alpha * (male_center - spider.position)

            spider.position = np.clip(spider.position, self.problem.lower_bounds, self.problem.upper_bounds)

    def _mating(self):
        males = [s for s in self.population if s.gender == 1]
        females = [s for s in self.population if s.gender == 0]
        if not males or not females: return

        dominant_male = max(males, key=lambda m: m.weight)

        radius = np.sum(self.problem.upper_bounds - self.problem.lower_bounds) / (2 * self.problem.dim)

        mating_group = [f for f in females if np.linalg.norm(dominant_male.position - f.position) < radius]
        if not mating_group: return

        mating_group.append(dominant_male)

        weights = np.array([s.weight for s in mating_group])
        probabilities = weights / np.sum(weights) if np.sum(weights) > 0 else None

        new_position = np.zeros(self.problem.dim)
        for dim in range(self.problem.dim):
            parent_idx = np.random.choice(len(mating_group), p=probabilities)
            new_position[dim] = mating_group[parent_idx].position[dim]

        new_fitness, _, _ = self.problem.evaluate(new_position)

        worst_spider = max(self.population, key=lambda s: s.fitness)
        if new_fitness < worst_spider.fitness:
            worst_spider.position = new_position
            worst_spider.fitness = new_fitness
