import numpy as np


def tournament(population, tournament_size=3, minimize=True):
    """
    Tournament selection: Select k random individuals and return the best.
    
    Args:
        population (list): List of Individual objects
        tournament_size (int): Number of individuals in tournament
        minimize (bool): True if minimizing fitness, False if maximizing
        
    Returns:
        Individual: Selected parent
    """
    # Select random individuals for tournament
    tournament_individuals = np.random.choice(population, tournament_size, replace=False)
    
    # Find the best individual in tournament
    if minimize:
        winner = min(tournament_individuals, key=lambda ind: ind.fitness)
    else:
        winner = max(tournament_individuals, key=lambda ind: ind.fitness)
    
    return winner