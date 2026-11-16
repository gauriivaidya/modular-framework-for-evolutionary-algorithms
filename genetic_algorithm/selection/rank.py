import numpy as np

def rank(population, minimize=True):
    """
    Rank-based selection.

    Individuals are ranked by fitness, and probabilities are proportional
    to rank (1..N). Works for both minimization and maximization.
    """
    if len(population) == 0:
        raise ValueError("Population is empty")

    if minimize:
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    else:
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)

    n = len(sorted_pop)
    ranks = np.arange(1, n + 1, dtype=float)
    probs = ranks / ranks.sum()

    idx = np.random.choice(n, p=probs)
    return sorted_pop[idx].copy()