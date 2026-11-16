import numpy as np
from genetic_algorithm.individual import Individual


def rbx(parent1, parent2):
    """
    RBX (Range-Based Crossover) for integer/discrete vectors.

    For each gene i:
      child[i] is sampled uniformly from [min(p1[i], p2[i]), max(p1[i], p2[i])] (inclusive)
    """
    p1 = np.asarray(parent1.chromosome, dtype=int)
    p2 = np.asarray(parent2.chromosome, dtype=int)

    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)

    child1_chrom = np.random.randint(lo, hi + 1, size=p1.shape, dtype=int)
    child2_chrom = np.random.randint(lo, hi + 1, size=p1.shape, dtype=int)

    child1 = Individual(parent1.representation_type, values=child1_chrom)
    child2 = Individual(parent2.representation_type, values=child2_chrom)
    return child1, child2