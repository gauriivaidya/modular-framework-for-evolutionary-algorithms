import numpy as np

def mpox(parent1, parent2, parent3):
    """
    3-parent MPOX (Multi-Parent Order Crossover) for permutation chromosomes.

    - Takes a segment from parent1
    - Fills remaining genes using order from parent2, then parent3
    - Child2: segment from parent2, fill from parent3 then parent1 (symmetry)

    Returns: (child1, child2) as Individual copies with updated chromosomes.
    """
    p1 = list(parent1.chromosome)
    p2 = list(parent2.chromosome)
    p3 = list(parent3.chromosome)

    n = len(p1)
    if n < 2 or len(p2) != n or len(p3) != n:
        return parent1.copy(), parent2.copy()

    # Ensure permutations (unique genes)
    if len(set(p1)) != n or len(set(p2)) != n or len(set(p3)) != n:
        return parent1.copy(), parent2.copy()

    a, b = sorted(np.random.choice(n, 2, replace=False))  # segment [a:b)
    # if you prefer inclusive, use b+1; here keep python slice semantics
    if a == b:
        return parent1.copy(), parent2.copy()

    def build_child(segment_donor, fill_first, fill_second):
        seg = segment_donor[a:b]
        seg_set = set(seg)

        child = [None] * n
        child[a:b] = seg

        fill = []
        used = set(seg)

        for gene in fill_first:
            if gene not in used:
                fill.append(gene)
                used.add(gene)
        for gene in fill_second:
            if gene not in used:
                fill.append(gene)
                used.add(gene)

        it = iter(fill)
        for i in range(n):
            if child[i] is None:
                child[i] = next(it)
        return child

    c1_list = build_child(p1, p2, p3)
    c2_list = build_child(p2, p3, p1)

    child1 = parent1.copy()
    child2 = parent2.copy()

    # Preserve chromosome container type
    if isinstance(parent1.chromosome, np.ndarray):
        child1.chromosome = np.asarray(c1_list, dtype=parent1.chromosome.dtype)
    else:
        child1.chromosome = c1_list

    if isinstance(parent2.chromosome, np.ndarray):
        child2.chromosome = np.asarray(c2_list, dtype=parent2.chromosome.dtype)
    else:
        child2.chromosome = c2_list

    child1.fitness = None
    child2.fitness = None
    return child1, child2