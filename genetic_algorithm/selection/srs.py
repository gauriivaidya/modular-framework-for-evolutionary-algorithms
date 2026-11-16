import numpy as np

def srs(population, minimize=True, lambda_plus=0.7, **_):
    """
    Split Rank Selection (SRS) - returns ONE selected individual (copy).
    Accepts `minimize` because GA passes it as a kwarg.
    """
    if len(population) == 0:
        raise ValueError("Population is empty")
    if not (0.0 <= lambda_plus <= 1.0):
        raise ValueError("lambda_plus must be in [0, 1]")

    K = len(population)
    half = K // 2
    lambda_minus = 1.0 - lambda_plus

    fitnesses = np.array([ind.fitness for ind in population], dtype=float)

    # Sort WORST -> BEST
    # minimize=True: worst has largest fitness => descending
    # minimize=False: worst has smallest fitness => ascending
    order = np.argsort(fitnesses)[::-1] if minimize else np.argsort(fitnesses)

    probs = np.zeros(K, dtype=float)
    for pos in range(K):
        i = pos + 1  # rank 1..K (worst..best)
        if i <= half:
            probs[pos] = lambda_minus * (8.0 * i) / (K * (K + 2))
        else:
            probs[pos] = lambda_plus  * (8.0 * i) / (K * (3 * K + 2))

    s = probs.sum()
    if s <= 0 or not np.isfinite(s):
        return population[np.random.randint(K)].copy()

    probs /= s
    pos = np.random.choice(K, p=probs)
    idx = order[pos]
    return population[idx].copy()