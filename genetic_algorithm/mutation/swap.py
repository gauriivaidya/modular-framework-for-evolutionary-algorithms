import numpy as np


def swap(individual):
    """
    Swap mutation: Swap two random positions.
    
    Best for: Permutation representations (maintains valid permutation).
    
    """
    mutated = individual.copy()
    
    # Select two random distinct positions
    idx1, idx2 = np.random.choice(len(mutated), 2, replace=False)
    
    # Swap the values
    mutated.chromosome[idx1], mutated.chromosome[idx2] = \
        mutated.chromosome[idx2], mutated.chromosome[idx1]
    
    return mutated