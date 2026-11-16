import numpy as np
from ..individual import Individual


def ox(parent1, parent2):
    """
    Order Crossover (OX) for permutation representations.    
   
    """
    size = len(parent1)
    point1, point2 = sorted(np.random.choice(range(1, size), 2, replace=False))
    
    def create_child(p1, p2):
        child = np.full(size, -1, dtype=p1.chromosome.dtype)
        # Copy middle from p1
        child[point1:point2] = p1.chromosome[point1:point2]
        
        # Fill rest from p2 in order
        p2_idx = point2
        child_idx = point2
        
        while -1 in child:
            if p2_idx >= size:
                p2_idx = 0
            if child_idx >= size:
                child_idx = 0
                
            if p2.chromosome[p2_idx] not in child:
                child[child_idx] = p2.chromosome[p2_idx]
                child_idx += 1
            p2_idx += 1
        
        return child
    
    child1_chromosome = create_child(parent1, parent2)
    child2_chromosome = create_child(parent2, parent1)
    
    child1 = Individual(parent1.representation_type, values=child1_chromosome)
    child2 = Individual(parent2.representation_type, values=child2_chromosome)
    
    return child1, child2