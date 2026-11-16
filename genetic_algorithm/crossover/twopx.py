import numpy as np
from ..individual import Individual


def twopx(parent1, parent2):
    """
    Two-Point Crossover (2PX) for vector representations (binary/int/real).
    Picks two cut points and swaps the middle segment between parents.
    """
    size = len(parent1)
    point1, point2 = sorted(np.random.choice(range(1, size), 2, replace=False))

    p1 = parent1.chromosome
    p2 = parent2.chromosome

    # Create children chromosomes
    child1_chromosome = p1.copy()
    child2_chromosome = p2.copy()

    # Swap the segment between point1 and point2
    child1_chromosome[point1:point2] = p2[point1:point2]
    child2_chromosome[point1:point2] = p1[point1:point2]

    # Wrap in Individuals (same style as your OX)
    child1 = Individual(parent1.representation_type, values=child1_chromosome)
    child2 = Individual(parent2.representation_type, values=child2_chromosome)

    return child1, child2