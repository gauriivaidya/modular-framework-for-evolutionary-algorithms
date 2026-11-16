import numpy as np
from genetic_algorithm.individual import Individual


def pmx(parent1, parent2):
    """
    Partially Mapped Crossover (PMX) for permutation chromosomes.

    Inputs:
      parent1, parent2: Individual
        - .chromosome is a permutation (list/np.ndarray) of unique genes

    Returns:
      (child1, child2): Individuals with permutation chromosomes
    """
    p1 = list(parent1.chromosome)
    p2 = list(parent2.chromosome)

    n = len(p1)
    if n != len(p2) or n < 2:
        return parent1.copy(), parent2.copy()

    # pick two cut points (inclusive segment)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    # We'll copy segment a..b (inclusive)
    seg1 = p1[a:b + 1]
    seg2 = p2[a:b + 1]

    # mapping between swapped segments
    map_1to2 = {seg1[i]: seg2[i] for i in range(len(seg1))}
    map_2to1 = {seg2[i]: seg1[i] for i in range(len(seg2))}

    def resolve(gene, mapping, segment_set):
        """Follow mapping until gene is not inside the swapped segment."""
        while gene in segment_set:
            gene = mapping.get(gene, gene)
        return gene

    # build children as lists
    c1 = [None] * n
    c2 = [None] * n

    # copy swapped segments
    c1[a:b + 1] = seg2
    c2[a:b + 1] = seg1

    seg2_set = set(seg2)
    seg1_set = set(seg1)

    # fill the rest from original parents, resolving conflicts via mapping
    for i in range(n):
        if i < a or i > b:
            g1 = p1[i]
            g2 = p2[i]
            c1[i] = resolve(g1, map_2to1, seg2_set)  # conflicts with seg2 in child1
            c2[i] = resolve(g2, map_1to2, seg1_set)  # conflicts with seg1 in child2

    # (optional) final sanity repair if something weird happened
    # ensure permutations by replacing duplicates with missing genes
    def repair(child):
        missing = [g for g in p1 if g not in child]
        seen = set()
        mi = 0
        for i, g in enumerate(child):
            if g in seen:
                child[i] = missing[mi]
                mi += 1
            else:
                seen.add(g)
        return child

    c1 = repair(c1)
    c2 = repair(c2)

    # return Individuals (preserve chromosome type)
    child1 = parent1.copy()
    child2 = parent2.copy()

    if isinstance(parent1.chromosome, np.ndarray):
        child1.chromosome = np.asarray(c1, dtype=parent1.chromosome.dtype)
    else:
        child1.chromosome = c1

    if isinstance(parent2.chromosome, np.ndarray):
        child2.chromosome = np.asarray(c2, dtype=parent2.chromosome.dtype)
    else:
        child2.chromosome = c2

    # fitness should be re-evaluated by GA after crossover
    child1.fitness = None
    child2.fitness = None

    return child1, child2