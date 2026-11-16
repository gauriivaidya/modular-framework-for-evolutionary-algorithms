import numpy as np
from core.optimizer import Optimizer


class Monkey:
    def __init__(self, num_dimensions, lower_bounds, upper_bounds):
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.fitness = float('inf')
        self.group_idx = -1
        self.ll_counter = 0


class SpiderMonkeyOptimization(Optimizer):
    def __init__(self, problem, population_size=50, num_generations=100,
                 num_groups=5, ll_limit=100, gl_limit=500, pr=0.1):
        super().__init__(problem, population_size, num_generations)
        self.num_groups = num_groups
        self.ll_limit = ll_limit
        self.gl_limit = gl_limit
        self.pr = pr
        self.gl_counter = 0

    def optimize(self):
        group_size = self.population_size // self.num_groups
        self.population = [Monkey(self.problem.dim, self.problem.lower_bounds, self.problem.upper_bounds) for _ in
                           range(self.population_size)]
        for i, monkey in enumerate(self.population):
            monkey.group_idx = i // group_size

        for gen in range(self.num_generations):
            for monkey in self.population:
                monkey.fitness, _, _ = self.problem.evaluate(monkey.position)

            local_leaders, global_leader = self._update_leaders()

            self._local_leader_phase(local_leaders)
            self._global_leader_phase(global_leader, local_leaders)
            self._local_leader_decision(local_leaders)
            self._global_leader_decision()

            self.logger.info(f"Generation {gen + 1}/{self.num_generations}, Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness

    def _update_leaders(self):
        local_leaders = []
        for i in range(self.num_groups):
            group = [m for m in self.population if m.group_idx == i]
            if group:
                leader = min(group, key=lambda m: m.fitness)
                local_leaders.append(leader)
                leader.ll_counter += 1

        global_leader = min(self.population, key=lambda m: m.fitness)
        if global_leader.fitness < self.gbest_fitness:
            self.gbest_fitness = global_leader.fitness
            self.gbest_position = global_leader.position.copy()
            self.gl_counter = 0
        else:
            self.gl_counter += 1
        return local_leaders, global_leader

    def _local_leader_phase(self, local_leaders):
        for monkey in self.population:
            if np.random.rand() >= self.pr:
                leader = local_leaders[monkey.group_idx]
                group = [m for m in self.population if m.group_idx == monkey.group_idx]
                random_monkey = group[np.random.randint(0, len(group))]

                update = np.random.rand() * (leader.position - monkey.position) + \
                         np.random.uniform(-1, 1) * (random_monkey.position - monkey.position)
                monkey.position += update
                monkey.position = np.clip(monkey.position, self.problem.lower_bounds, self.problem.upper_bounds)

    def _global_leader_phase(self, global_leader, local_leaders):
        for monkey in self.population:
            prob = 0.9 * (monkey.fitness / (self.gbest_fitness + 1e-100)) + 0.1
            if np.random.rand() < prob:
                group = [m for m in self.population if m.group_idx == monkey.group_idx]
                if not group: continue
                random_monkey = group[np.random.randint(0, len(group))]

                dim_to_update = np.random.randint(0, self.problem.dim)

                update_component_1 = np.random.rand() * (
                            global_leader.position[dim_to_update] - monkey.position[dim_to_update])
                update_component_2 = np.random.uniform(-1, 1) * (
                            random_monkey.position[dim_to_update] - monkey.position[dim_to_update])

                monkey.position[dim_to_update] += update_component_1 + update_component_2
                monkey.position = np.clip(monkey.position, self.problem.lower_bounds, self.problem.upper_bounds)

    def _local_leader_decision(self, local_leaders):
        for i in range(self.num_groups):
            if i >= len(local_leaders): continue
            leader = local_leaders[i]
            if leader.ll_counter > self.ll_limit:
                leader.ll_counter = 0
                group_monkeys = [m for m in self.population if m.group_idx == leader.group_idx]
                for monkey in group_monkeys:
                    r1 = np.random.rand()
                    if r1 >= self.pr:
                        monkey.position = self.problem.lower_bounds + np.random.rand(self.problem.dim) * (
                                    self.problem.upper_bounds - self.problem.lower_bounds)
                    else:
                        r2 = np.random.rand()
                        monkey.position += r2 * (self.gbest_position - monkey.position) - r2 * (
                                    leader.position - monkey.position)
                    monkey.position = np.clip(monkey.position, self.problem.lower_bounds, self.problem.upper_bounds)

    def _global_leader_decision(self):
        if self.gl_counter > self.gl_limit:
            self.gl_counter = 0
            if self.num_groups < self.population_size // 2:  # Max groups limit
                group_sizes = [len([m for m in self.population if m.group_idx == i]) for i in range(self.num_groups)]
                if not group_sizes: return
                largest_group_idx = np.argmax(group_sizes)
                self.num_groups += 1
                group_to_split = [m for m in self.population if m.group_idx == largest_group_idx]
                for i, monkey in enumerate(group_to_split):
                    if i >= len(group_to_split) / 2:
                        monkey.group_idx = self.num_groups - 1
            else:
                self.num_groups = 1
                for monkey in self.population:
                    monkey.group_idx = 0


