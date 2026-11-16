import numpy as np


class TSPProblem:
    """
    TSP:
      - chromosome is a permutation of cities [0..N-1]
      - objective: minimize tour length visiting all cities once and returning to start
    """

    def __init__(self, n_cities, coords, name=None):
        self.n_cities = int(n_cities)

        self.size = self.n_cities
        self.dimension = self.n_cities
        self.num_dimensions = self.n_cities
        self.is_minimization = True
        self.name = name or f"tsp-{self.n_cities}"

        self.coords = np.asarray(coords, dtype=float)

        if self.coords.shape != (self.n_cities, 2):
            raise ValueError(f"coords must be shape ({self.n_cities}, 2)")

        # Precompute distance matrix
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist = np.sqrt(np.sum(diff * diff, axis=2))

        # Compatibility aliases for ACO
        self.distance_matrix = self.dist
        self.distances = self.dist
        self.coordinates = self.coords

    def evaluate(self, chromosome):
        tour = list(chromosome)

        if len(tour) != self.n_cities:
            raise ValueError(f"chromosome length must be {self.n_cities}")

        total = 0.0

        for i in range(self.n_cities):
            a = int(tour[i])
            b = int(tour[(i + 1) % self.n_cities])
            total += self.dist[a, b]

        return float(total)

    def initial_state(self):
        return [0]

    def feasible_moves(self, state):
        visited = set(state)
        return [city for city in range(self.n_cities) if city not in visited]

    def apply_move(self, state, move):
        return state + [int(move)]

    def is_complete(self, state):
        return len(state) == self.n_cities

    def finalize_solution(self, state):
        return list(state)

    def pheromone_key(self, state, move):
        current_city = state[-1]
        next_city = int(move)
        return current_city, next_city

    def all_components(self):
        return [
            (i, j)
            for i in range(self.n_cities)
            for j in range(self.n_cities)
            if i != j
        ]

    def solution_components(self, solution):
        tour = list(solution)
        components = []

        for i in range(len(tour)):
            a = int(tour[i])
            b = int(tour[(i + 1) % len(tour)])
            components.append((a, b))

        return components

    def heuristic_value(self, state, move):
        current_city = state[-1]
        next_city = int(move)
        distance = self.dist[current_city, next_city]

        return 1.0 / (distance + 1e-10)

    def solution_to_array(self, solution):
        return np.array(solution, dtype=int)

    def neighbors(self, solution):
        tour = list(solution)
        n = len(tour)

        for i in range(1, n - 2):
            for j in range(i + 1, n):
                candidate = tour[:]
                candidate[i:j] = reversed(candidate[i:j])
                yield candidate

    def candidate_moves(self, state, moves, candidate_k=10):
        current_city = state[-1]
        move_set = set(moves)

        ordered = np.argsort(self.dist[current_city])

        candidates = [
            int(city)
            for city in ordered
            if city != current_city and int(city) in move_set
        ]

        return candidates[:candidate_k] if candidates else list(moves)


def create_tsp_problem(
    n_cities=31,
    seed=69,
    coord_range=(0.0, 100.0),
    name=None
):
    """
    Factory: random 2D Euclidean TSP instance.
    """
    rng = np.random.default_rng(seed)

    low = float(coord_range[0])
    high = float(coord_range[1])

    coords = rng.uniform(low, high, size=(n_cities, 2))

    return TSPProblem(
        n_cities=n_cities,
        coords=coords,
        name=name or f"tsp-{n_cities}"
    )


# Optional default instance for your fallback loader
tsp_problem = create_tsp_problem()