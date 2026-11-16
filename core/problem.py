import numpy as np


class OptimizationProblem:
    """
    A class to represent an optimization problem.

    This class encapsulates all the necessary information about an optimization
    problem, including the objective function, variable bounds, and the optimization
    goal (minimization or maximization).
    """

    def __init__(self, name, objective_function, lower_bounds, upper_bounds, is_minimization=True):
        """
        Initializes an OptimizationProblem object.

        Args:
            name (str): The name of the optimization problem.
            objective_function (callable): The objective function to be optimized.
                                           It should take a NumPy array of variables
                                           as input and return a single fitness value.
            lower_bounds (list or np.ndarray): A list or NumPy array of the lower
                                               bounds for each variable.
            upper_bounds (list or np.ndarray): A list or NumPy array of the upper
                                               bounds for each variable.
            is_minimization (bool, optional): A flag indicating whether the problem
                                              is a minimization problem. Defaults to True.
        """
        self.name = name
        self.objective_function = objective_function
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.is_minimization = is_minimization
        self.num_dimensions = len(lower_bounds)

    def evaluate(self, position):
        """
        Evaluates the objective function for a given position.

        Args:
            position (np.ndarray): A NumPy array representing the position (i.e.,
                                   the set of variable values) to be evaluated.

        Returns:
            float: The fitness value of the given position.
        """
        return self.objective_function(position)
