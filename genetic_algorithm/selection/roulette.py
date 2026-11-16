import numpy as np

def roulette(population, minimize=True):
    """
    Roulette wheel (fitness proportionate) selection.

    For minimization problems, we transform fitness so that smaller fitness
    => larger selection probability.
    """
    if len(population) == 0:
        raise ValueError("Population is empty")

    fitnesses = np.array([ind.fitness for ind in population], dtype=float)

    # Transform fitness to "scores" where larger is better
    if minimize:
        # Smaller fitness => better => larger score
        max_fit = fitnesses.max()
        scores = max_fit - fitnesses + 1e-12
    else:
        # Larger fitness => better => larger score
        min_fit = fitnesses.min()
        scores = fitnesses - min_fit + 1e-12

    # If all scores are (almost) equal or non-positive, fallback to random
    total = scores.sum()
    if total <= 0 or not np.isfinite(total):
        idx = np.random.randint(len(population))
        return population[idx].copy()

    probs = scores / total
    idx = np.random.choice(len(population), p=probs)
    return population[idx].copy()
