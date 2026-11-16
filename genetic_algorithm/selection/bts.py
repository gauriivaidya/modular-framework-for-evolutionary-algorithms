import numpy as np

def bts(population, minimize=True, **_):
    """
    bts is a binary version of tournament selection
    """
    if len(population) == 0:
        raise ValueError("Population is empty")

    i, j = np.random.randint(0, len(population), size=2)
    a, b = population[i], population[j]

    if minimize:
        winner = a if a.fitness <= b.fitness else b
    else:
        winner = a if a.fitness >= b.fitness else b

    return winner.copy()