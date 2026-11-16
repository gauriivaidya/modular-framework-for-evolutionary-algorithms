from pathlib import Path
import math

import numpy as np


class VRPProblem:

    def __init__(
        self,
        customer_coords,
        demands,
        depot,
        vehicle_capacity,
        max_route_time=0.0,
        drop_time=0.0,
        name="vrp"
    ):
        self.customer_coords = np.asarray(
            customer_coords,
            dtype=float
        )

        self.demands = np.asarray(
            demands,
            dtype=float
        )

        self.depot = np.asarray(
            depot,
            dtype=float
        )

        self.vehicle_capacity = float(
            vehicle_capacity
        )

        self.max_route_time = float(
            max_route_time
        )

        self.drop_time = float(
            drop_time
        )

        self.num_customers = len(
            self.customer_coords
        )

        self.name = name
        self.is_minimization = True

        self.size = self.num_customers
        self.dimension = self.num_customers
        self.num_dimensions = self.num_customers

        self.distance_matrix = (
            self._build_distance_matrix()
        )

    # =====================================================
    # Distances
    # =====================================================

    def _build_distance_matrix(self):
        """
        Index 0 = depot
        Index i+1 = customer i
        """

        points = np.vstack([
            self.depot,
            self.customer_coords
        ])

        n = len(points)

        matrix = np.zeros(
            (n, n),
            dtype=float
        )

        for i in range(n):
            for j in range(n):

                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]

                matrix[i, j] = math.hypot(
                    dx,
                    dy
                )

        return matrix

    # =====================================================
    # Solution decoding
    # =====================================================

    def _decode_routes(self, solution):
        """
        Solution is a customer permutation.

        We split the permutation into routes greedily
        according to vehicle capacity.
        """

        routes = []

        current_route = []
        current_load = 0.0

        for customer in solution:

            customer = int(customer)
            demand = self.demands[customer]

            if (
                current_route
                and current_load + demand
                > self.vehicle_capacity
            ):
                routes.append(
                    current_route
                )

                current_route = []
                current_load = 0.0

            current_route.append(
                customer
            )

            current_load += demand

        if current_route:
            routes.append(
                current_route
            )

        return routes

    # =====================================================
    # Objective
    # =====================================================

    def evaluate(self, solution):

        if len(solution) != self.num_customers:
            raise ValueError(
                f"Expected {self.num_customers} customers, "
                f"got {len(solution)}"
            )

        if len(set(solution)) != self.num_customers:
            return float("inf")

        routes = self._decode_routes(
            solution
        )

        total_distance = 0.0

        for route in routes:

            if not route:
                continue

            # Depot -> first customer
            first = route[0]

            total_distance += (
                self.distance_matrix[
                    0,
                    first + 1
                ]
            )

            # Customer -> customer
            for i in range(
                len(route) - 1
            ):
                a = route[i]
                b = route[i + 1]

                total_distance += (
                    self.distance_matrix[
                        a + 1,
                        b + 1
                    ]
                )

            # Last customer -> depot
            last = route[-1]

            total_distance += (
                self.distance_matrix[
                    last + 1,
                    0
                ]
            )

            # Optional maximum-route-time constraint
            if self.max_route_time > 0:

                route_time = 0.0

                route_time += (
                    self.distance_matrix[
                        0,
                        first + 1
                    ]
                )

                for i in range(
                    len(route) - 1
                ):
                    route_time += (
                        self.distance_matrix[
                            route[i] + 1,
                            route[i + 1] + 1
                        ]
                    )

                route_time += (
                    self.distance_matrix[
                        last + 1,
                        0
                    ]
                )

                route_time += (
                    self.drop_time
                    * len(route)
                )

                if route_time > self.max_route_time:
                    return float("inf")

        return float(
            total_distance
        )

    # =====================================================
    # ACO construction
    # =====================================================

    def initial_state(self):
        return []

    def feasible_moves(self, state):
        """
        Any customer not yet visited can be selected.
        """

        if len(state) >= self.num_customers:
            return []

        visited = set(state)

        return [
            customer
            for customer
            in range(self.num_customers)
            if customer not in visited
        ]

    def apply_move(
        self,
        state,
        move
    ):
        return state + [int(move)]

    def is_complete(
        self,
        state
    ):
        return (
            len(state)
            == self.num_customers
        )

    def finalize_solution(
        self,
        state
    ):
        return list(state)

    # =====================================================
    # Pheromone
    # =====================================================

    def pheromone_key(
        self,
        state,
        move
    ):
        """
        Edge pheromone.

        -1 represents the depot.
        """

        if len(state) == 0:
            previous = -1
        else:
            previous = int(
                state[-1]
            )

        return (
            previous,
            int(move)
        )

    def all_components(self):

        components = []

        # Depot -> customer
        for customer in range(
            self.num_customers
        ):
            components.append(
                (-1, customer)
            )

        # Customer -> customer
        for i in range(
            self.num_customers
        ):
            for j in range(
                self.num_customers
            ):
                if i != j:
                    components.append(
                        (i, j)
                    )

        return components

    def solution_components(
        self,
        solution
    ):

        components = []

        if not solution:
            return components

        components.append(
            (-1, int(solution[0]))
        )

        for i in range(
            len(solution) - 1
        ):
            components.append(
                (
                    int(solution[i]),
                    int(solution[i + 1])
                )
            )

        return components

    # =====================================================
    # Heuristic
    # =====================================================

    def heuristic_value(
        self,
        state,
        move
    ):
        """
        Standard VRP/TSP heuristic:
        inverse distance to next customer.
        """

        customer = int(move)

        if len(state) == 0:

            distance = (
                self.distance_matrix[
                    0,
                    customer + 1
                ]
            )

        else:

            previous = int(
                state[-1]
            )

            distance = (
                self.distance_matrix[
                    previous + 1,
                    customer + 1
                ]
            )

        return 1.0 / max(
            distance,
            1e-12
        )

    # =====================================================
    # Helpers
    # =====================================================

    def solution_to_array(
        self,
        solution
    ):
        return np.asarray(
            solution,
            dtype=int
        )

    def neighbors(
        self,
        solution
    ):
        """
        Swap neighborhood.
        """

        solution = list(solution)

        for i in range(
            self.num_customers
        ):
            for j in range(
                i + 1,
                self.num_customers
            ):

                neighbor = solution[:]

                neighbor[i], neighbor[j] = (
                    neighbor[j],
                    neighbor[i]
                )

                yield neighbor


# =========================================================
# OR-Library loader
# =========================================================

def load_vrp_instance(filepath):

    filepath = Path(filepath)

    tokens = (
        filepath
        .read_text()
        .split()
    )

    pos = 0

    num_customers = int(
        tokens[pos]
    )

    vehicle_capacity = float(
        tokens[pos + 1]
    )

    max_route_time = float(
        tokens[pos + 2]
    )

    drop_time = float(
        tokens[pos + 3]
    )

    pos += 4

    depot_x = float(
        tokens[pos]
    )

    depot_y = float(
        tokens[pos + 1]
    )

    pos += 2

    depot = (
        depot_x,
        depot_y
    )

    customer_coords = []
    demands = []

    for _ in range(
        num_customers
    ):

        x = float(
            tokens[pos]
        )

        y = float(
            tokens[pos + 1]
        )

        demand = float(
            tokens[pos + 2]
        )

        pos += 3

        customer_coords.append(
            (x, y)
        )

        demands.append(
            demand
        )

    return (
        customer_coords,
        demands,
        depot,
        vehicle_capacity,
        max_route_time,
        drop_time
    )


# =========================================================
# Factory
# =========================================================

def create_vrp_problem(
    instance
):

    root = (
        Path(__file__).resolve()
        .parent.parent
        / "data"
        / "or_library"
        / "vrp"
    )

    filepath = root / instance

    if not filepath.exists():

        raise FileNotFoundError(
            f"VRP instance not found: "
            f"{filepath}"
        )

    (
        customer_coords,
        demands,
        depot,
        vehicle_capacity,
        max_route_time,
        drop_time
    ) = load_vrp_instance(
        filepath
    )

    return VRPProblem(
        customer_coords=customer_coords,
        demands=demands,
        depot=depot,
        vehicle_capacity=vehicle_capacity,
        max_route_time=max_route_time,
        drop_time=drop_time,
        name=filepath.stem
    )