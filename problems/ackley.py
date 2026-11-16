import numpy as np
from core.problem import OptimizationProblem
from core.utils import data_loader


def create_ackley_problem(dimensions=10):
    """
    Creates an instance of the Ackley benchmark problem.

    Args:
        dimensions (int): The number of dimensions for the problem.

    Returns:
        OptimizationProblem: An instance of the Ackley problem.
    """

    def ackley_objective(x):
        """
        Objective function for the Ackley benchmark problem.
        """
        # Load data using the DataLoader
        o_data = data_loader.load_mat_data('ackley_shift_func_data.mat')['o'][0]
        fbias_data = data_loader.load_mat_data('fbias_data.mat')['f_bias'][0]

        o = o_data[:dimensions]
        fbias = fbias_data[5]  # fbias for Ackley is at index 5

        # Shift the input vector
        z = x - o

        # Calculate the two main components of the Ackley function
        sum1 = np.sum(z ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * z))

        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / dimensions))
        term2 = -np.exp(sum2 / dimensions)

        obj = term1 + term2 + 20 + np.e + fbias

        # This is a benchmark function, so penalty is 0
        penalty = 0

        return obj, obj, penalty

    lower_bounds = [-32] * dimensions
    upper_bounds = [32] * dimensions

    return OptimizationProblem(
        name=f"Ackley-{dimensions}D",
        objective_function=ackley_objective,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        is_minimization=True
    )
