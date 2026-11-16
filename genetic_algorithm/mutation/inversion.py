import numpy as np

def inversion(individual):
    """
    Inversion mutation:
    - Pick two cut points i < j
    - Reverse the segment chromosome[i:j+1]
    """
    mutant = individual.copy()
    size = len(mutant.chromosome)

    if size < 2:
        return mutant

    i, j = np.random.choice(size, size=2, replace=False)
    if i > j:
        i, j = j, i

    mutant.chromosome[i:j + 1] = mutant.chromosome[i:j + 1][::-1]
    return mutant
