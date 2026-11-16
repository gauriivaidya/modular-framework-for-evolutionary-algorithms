import numpy as np

def nts(population, minimize=True, **_):
    """
    NTS rank-based selection using:
        P_i = 2(3i - 1) / (K(3K + 1))
    where i = 1..K ranks individuals from WORST -> BEST.

    Returns ONE selected individual (copy), 
    """
    if len(population) == 0:
        raise ValueError("Population is empty")

    K = len(population)
    fitnesses = np.array([ind.fitness for ind in population], dtype=float)

    # Sort WORST -> BEST
    # minimize=True: worst has largest fitness => descending
    # minimize=False: worst has smallest fitness => ascending
    order = np.argsort(fitnesses)[::-1] if minimize else np.argsort(fitnesses)

    # Probabilities by rank position (pos=0..K-1 corresponds to rank i=1..K)
    probs = np.empty(K, dtype=float)
    denom = K * (3 * K + 1)
    for pos in range(K):
        i = pos + 1
        probs[pos] = 2.0 * (3 * i - 1) / denom

    # (This distribution already sums to 1, but keep a safety fallback)
    s = probs.sum()
    if s <= 0 or not np.isfinite(s):
        return population[np.random.randint(K)].copy()

    # sample a rank-position, then map back to the original index
    pos = np.random.choice(K, p=probs)
    idx = order[pos]
    return population[idx].copy()