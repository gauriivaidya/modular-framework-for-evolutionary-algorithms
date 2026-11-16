#OneMax Problem - Simple discrete optimization problem for testing GA.

class OneMaxProblem:
    """
    OneMax: Count the number of 1s in a binary string.
    Simple problem to test GA with binary representation.
    """
    
    def __init__(self, size=20):

        self.size = size
        self.dimension = size  
        self.num_dimensions = size
        self.name = f"one_max-{size}"
        self.is_minimization = False 
    
    def evaluate(self, chromosome):
        #Evaluate fitness: count number of 1s.
        fitness = sum(chromosome)
        return fitness


def create_one_max_problem(size=20):

    return OneMaxProblem(size)


# Default instance
one_max_problem = create_one_max_problem()
