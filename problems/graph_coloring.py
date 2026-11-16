from pathlib import Path

import numpy as np


class GraphColoringProblem:

    def __init__(
        self,
        adjacency_matrix,
        max_colors=None,
        conflict_penalty=1000.0,
        name="graph_coloring"
    ):
        self.adjacency_matrix = np.asarray(
            adjacency_matrix,
            dtype=int
        )

        self.n_vertices = self.adjacency_matrix.shape[0]

        # Allow up to n colors in the worst case
        if max_colors is None:
            max_colors = self.n_vertices

        self.max_colors = int(max_colors)
        self.conflict_penalty = float(conflict_penalty)

        self.name = name

        self.size = self.n_vertices
        self.dimension = self.n_vertices
        self.num_dimensions = self.n_vertices

        self.is_minimization = True

    # =====================================================
    # Evaluation
    # =====================================================

    def evaluate(self, solution):

        colors = np.asarray(
            solution,
            dtype=int
        )

        if len(colors) != self.n_vertices:
            raise ValueError(
                f"Expected {self.n_vertices} colors, "
                f"got {len(colors)}"
            )

        conflicts = 0

        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):

                if (
                    self.adjacency_matrix[i, j] == 1
                    and colors[i] == colors[j]
                ):
                    conflicts += 1

        used_colors = len(
            set(colors.tolist())
        )

        return float(
            used_colors
            + self.conflict_penalty * conflicts
        )

    # =====================================================
    # ACO construction
    # =====================================================

    def initial_state(self):
        return []

    def feasible_moves(self, state):

        vertex = len(state)

        if vertex >= self.n_vertices:
            return []

        # First vertex always uses color 0
        if vertex == 0:
            return [0]

        used_colors = sorted(set(state))

        forbidden_colors = set()

        for previous_vertex, previous_color in enumerate(state):

            if (
                self.adjacency_matrix[
                    vertex,
                    previous_vertex
                ] == 1
            ):
                forbidden_colors.add(
                    previous_color
                )

        # Existing legal colors
        moves = [
            color
            for color in used_colors
            if color not in forbidden_colors
        ]

        # Allow one new color
        new_color = max(used_colors) + 1

        if new_color < self.max_colors:
            moves.append(new_color)

        return moves

    def apply_move(self, state, move):
        return state + [int(move)]

    def is_complete(self, state):
        return len(state) == self.n_vertices

    def finalize_solution(self, state):
        return list(state)

    # =====================================================
    # Pheromone
    # =====================================================

    def pheromone_key(self, state, move):

        vertex = len(state)

        return (
            vertex,
            int(move)
        )

    def all_components(self):

        return [
            (vertex, color)
            for vertex in range(self.n_vertices)
            for color in range(self.max_colors)
        ]

    def solution_components(self, solution):

        return [
            (
                vertex,
                int(color)
            )
            for vertex, color
            in enumerate(solution)
        ]

    # =====================================================
    # Heuristic
    # =====================================================

    def heuristic_value(self, state, move):

        if len(state) == 0:
            return 1.0

        color = int(move)

        # Prefer reusing a color rather than
        # opening a new one
        if color in state:
            return 2.0

        return 1.0

    # =====================================================
    # Helpers
    # =====================================================

    def solution_to_array(self, solution):

        return np.asarray(
            solution,
            dtype=int
        )

    def neighbors(self, solution):

        solution = list(solution)

        for vertex in range(self.n_vertices):

            current_color = solution[vertex]

            for color in range(self.max_colors):

                if color == current_color:
                    continue

                neighbor = solution[:]
                neighbor[vertex] = color

                yield neighbor


# =========================================================
# OR-Library / DIMACS loader
# =========================================================

def load_graph_coloring_instance(filepath):

    filepath = Path(filepath)

    num_vertices = None
    expected_edges = None

    edges = []

    with filepath.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Comment
            if line.startswith("c"):
                continue

            parts = line.split()

            # Example:
            # p edge 100 500
            if parts[0] == "p":

                num_vertices = int(
                    parts[2]
                )

                expected_edges = int(
                    parts[3]
                )

            # Example:
            # e 1 2
            elif parts[0] == "e":

                u = int(parts[1]) - 1
                v = int(parts[2]) - 1

                edges.append(
                    (u, v)
                )

    if num_vertices is None:
        raise ValueError(
            f"No DIMACS 'p' line found in {filepath}"
        )

    if (
        expected_edges is not None
        and len(edges) != expected_edges
    ):
        raise ValueError(
            f"Expected {expected_edges} edges, "
            f"but parsed {len(edges)}"
        )

    # ---------------------------------------------
    # Convert edges into adjacency matrix
    # ---------------------------------------------

    adjacency_matrix = np.zeros(
        (
            num_vertices,
            num_vertices
        ),
        dtype=int
    )

    for u, v in edges:

        adjacency_matrix[u, v] = 1
        adjacency_matrix[v, u] = 1

    return adjacency_matrix


# =========================================================
# Factory
# =========================================================

def create_graph_coloring_problem(
    instance,
    max_colors=None,
    conflict_penalty=1000.0
):

    root = (
        Path(__file__).resolve()
        .parent.parent
        / "data"
        / "or_library"
        / "graph_coloring"
    )

    filepath = root / instance

    if not filepath.exists():
        raise FileNotFoundError(
            f"Graph Coloring instance not found: {filepath}"
        )

    adjacency_matrix = load_graph_coloring_instance(
        filepath
    )

    return GraphColoringProblem(
        adjacency_matrix=adjacency_matrix,
        max_colors=max_colors,
        conflict_penalty=conflict_penalty,
        name=filepath.stem
    )