import numpy as np


def mppmx(parent1, parent2, parent3):
    """
    3-parent MPPMX (Multi-Parent PMX) for permutation chromosomes.

    Idea (practical, robust variant):
      - Take a PMX segment from a "base" parent.
      - Use a second parent as the main PMX donor for the rest (with mapping repair).
      - If any holes remain (rare), fill with the third parent's order (fallback).

    Returns: (child1, child2) as Individual copies.
    """
    p1 = list(parent1.chromosome)
    p2 = list(parent2.chromosome)
    p3 = list(parent3.chromosome)
    n = len(p1)

    if n < 2 or len(p2) != n or len(p3) != n:
        return parent1.copy(), parent2.copy()

    # must be permutations
    if len(set(p1)) != n or len(set(p2)) != n or len(set(p3)) != n:
        return parent1.copy(), parent2.copy()

    a, b = sorted(np.random.choice(n, 2, replace=False))
    # inclusive segment [a..b]
    a, b = int(a), int(b)

    def build_child(base, donor, fallback):
        seg_base = base[a:b+1]
        seg_donor = donor[a:b+1]

        # mapping between base segment and donor segment
        # (used to resolve duplicates when placing donor genes outside segment)
        map_donor_to_base = {seg_donor[i]: seg_base[i] for i in range(len(seg_base))}

        child = [None] * n
        child[a:b+1] = seg_base
        seg_set = set(seg_base)

        def resolve(g):
            # if gene conflicts (already in segment), follow mapping until it doesn't
            while g in seg_set and g in map_donor_to_base:
                g = map_donor_to_base[g]
            return g

        # fill outside segment using donor with PMX conflict resolution
        for i in range(n):
            if i < a or i > b:
                g = resolve(donor[i])
                child[i] = g

        # final repair: if duplicates still exist (can happen in some edge mappings),
        # fill remaining/duplicate slots using fallback parent's order.
        seen = set()
        dup_positions = []
        for i, g in enumerate(child):
            if g in seen:
                dup_positions.append(i)
            else:
                seen.add(g)

        missing = [g for g in base if g not in seen]
        if not missing:
            # if base had all genes, still might need to fill None
            missing = [g for g in fallback if g not in seen]

        # Also handle any None positions (shouldn't happen often)
        for i, g in enumerate(child):
            if g is None:
                dup_positions.append(i)

        # Fill duplicates/Nones with missing genes, in fallback order preference
        # (missing list is already "base missing", which is fine)
        mi = 0
        for pos in dup_positions:
            if mi < len(missing):
                child[pos] = missing[mi]
                mi += 1

        # If still None (extremely unlikely), fill from fallback remaining
        if any(g is None for g in child):
            left = [g for g in fallback if g not in child]
            it = iter(left)
            for i in range(n):
                if child[i] is None:
                    child[i] = next(it)

        return child

    c1_list = build_child(p1, p2, p3)
    c2_list = build_child(p2, p3, p1)

    child1 = parent1.copy()
    child2 = parent2.copy()

    # preserve container type
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