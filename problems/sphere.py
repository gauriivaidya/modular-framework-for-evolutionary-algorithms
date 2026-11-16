import numpy as np

class sphere:
    """Minimize sum(x_i^2). Global optimum at all zeros."""
    def __init__(self, dim=20, lower=-50, upper=50, name=None):
        self.dim = int(dim)
        self.lower = float(lower)
        self.upper = float(upper)

        self.size = self.dim
        self.dimension = self.dim
        self.num_dimensions = self.dim
        self.is_minimization = True
        self.name = name or f"sphere-{self.dim}"

    def evaluate(self, chromosome):
        x = np.asarray(chromosome, dtype=float)
        return float(np.sum(x * x))

def create_sphere_problem(dim=20, lower=-50, upper=50):
    return sphere(dim=dim, lower=lower, upper=upper)