import numpy as np
from ..individual import Individual

def cx(parent1, parent2):
    """
    Cycle crossover (CX) for permutation chromosomes.

    Produces two children that are valid permutations of the parents' values.
    """
    p1 = parent1.chromosome
    p2 = parent2.chromosome

    if len(p1) != len(p2):
        raise ValueError("Parents must have the same chromosome length")

    size = len(p1)

    # Initialize children as "empty"
    child1 = np.full(size, -1, dtype=p1.dtype)
    child2 = np.full(size, -1, dtype=p1.dtype)

    visited = np.zeros(size, dtype=bool)
    cycle = 0

    while not np.all(visited):
        # Start a new cycle from the first unvisited index
        start_idx = np.where(~visited)[0][0]
        idx = start_idx
        cycle_indices = []

        # Follow the cycle
        while not visited[idx]:
            visited[idx] = True
            cycle_indices.append(idx)
            val_in_p2 = p2[idx]
            # Find position of this value in p1
            idx = np.where(p1 == val_in_p2)[0][0]

        cycle_indices = np.array(cycle_indices, dtype=int)

        if cycle % 2 == 0:
            # Even cycle: child1 gets genes from parent1, child2 from parent2
            child1[cycle_indices] = p1[cycle_indices]
            child2[cycle_indices] = p2[cycle_indices]
        else:
            # Odd cycle: swap
            child1[cycle_indices] = p2[cycle_indices]
            child2[cycle_indices] = p1[cycle_indices]

        cycle += 1

    child1_ind = Individual(
        representation_type=parent1.representation_type,
        values=child1
    )
    child2_ind = Individual(
        representation_type=parent2.representation_type,
        values=child2
    )

    return child1_ind, child2_ind
