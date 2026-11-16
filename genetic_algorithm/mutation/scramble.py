import numpy as np

def scramble(individual):
    """
    Scramble mutation:
    - Pick two cut points i < j
    - Randomly shuffle the elements inside chromosome[i:j+1]
    """
    mutant = individual.copy()
    size = len(mutant.chromosome)

    if size < 2:
        return mutant

    i, j = np.random.choice(size, size=2, replace=False)
    if i > j:
        i, j = j, i

    segment = mutant.chromosome[i:j + 1].copy()
    np.random.shuffle(segment)
    mutant.chromosome[i:j + 1] = segment
    return mutant
