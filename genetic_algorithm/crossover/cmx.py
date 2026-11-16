import numpy as np


def cmx(parent1, parent2):
    """
    Completely Mapped Crossover (CMX) for permutation chromosomes.

    Expects:
      parent.chromosome is a permutation (list/np.ndarray) of unique items.

    Returns:
      (child1, child2) as Individual copies with updated chromosomes.
    """

    p1 = list(parent1.chromosome)
    p2 = list(parent2.chromosome)

    n = len(p1)
    if n != len(p2) or n == 0:
        return parent1.copy(), parent2.copy()

    # If parents are identical, nothing to do
    start_idx = next((i for i in range(n) if p1[i] != p2[i]), None)
    if start_idx is None:
        return parent1.copy(), parent2.copy()

    # CMX assumes permutations; we still try to be safe
    pos_in_p1 = {gene: i for i, gene in enumerate(p1)}

    # --- Build cut-box rows C1 and C2 (cycle-based) ---
    # Step 1: C1 starts with the mismatching gene from parent2
    c1 = [p2[start_idx]]
    c2 = []

    # Steps 2–4: alternate constructing C2 then C1 until the cycle closes
    for _ in range(n):
        v = c1[-1]
        if v not in pos_in_p1:
            break

        # Step 2 (condensed): v -> (under v in P2) = u -> (under u in P2) = w; append w to C2
        u = p2[pos_in_p1[v]]
        if u not in pos_in_p1:
            break
        w = p2[pos_in_p1[u]]

        if w in c2:
            break
        c2.append(w)

        # Step 3: find w in P1, take the parallel gene from P2 as next C1 entry
        x = c2[-1]
        if x not in pos_in_p1:
            break
        y = p2[pos_in_p1[x]]

        if y in c1:
            break
        c1.append(y)

    cut1 = set(c1)
    cut2 = set(c2) if c2 else set(c1)  # fallback, should rarely happen

    # --- Step 6: offspring templates (keep non-cut genes in-place) ---
    o1 = [None] * n
    o2 = [None] * n
    holes1 = []
    holes2 = []

    for i in range(n):
        if p1[i] in cut1:
            holes1.append(i)
        else:
            o1[i] = p1[i]

        if p2[i] in cut2:
            holes2.append(i)
        else:
            o2[i] = p2[i]

    holes1.sort()
    holes2.sort()

    # --- Step 7: map C2 over O1 and C1 over O2 (fill holes in index order) ---
    fill1 = list(c2)
    fill2 = list(c1)

    # Safety fallback if something odd happened (e.g., non-permutation input)
    if len(fill1) != len(holes1):
        missing = [g for g in p2 if g not in o1]
        fill1 = (fill1 + missing)[: len(holes1)]
    if len(fill2) != len(holes2):
        missing = [g for g in p1 if g not in o2]
        fill2 = (fill2 + missing)[: len(holes2)]

    for k, idx in enumerate(holes1):
        o1[idx] = fill1[k]
    for k, idx in enumerate(holes2):
        o2[idx] = fill2[k]

    # Build children as copies of parents (keeps metadata/fitness handling consistent with your framework)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Preserve chromosome type (np.ndarray vs list)
    if isinstance(parent1.chromosome, np.ndarray):
        child1.chromosome = np.asarray(o1, dtype=parent1.chromosome.dtype)
    else:
        child1.chromosome = o1

    if isinstance(parent2.chromosome, np.ndarray):
        child2.chromosome = np.asarray(o2, dtype=parent2.chromosome.dtype)
    else:
        child2.chromosome = o2

    return child1, child2