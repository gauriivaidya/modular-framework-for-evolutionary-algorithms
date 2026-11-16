import numpy as np
from core.problem import OptimizationProblem
from core.utils import data_loader


def create_rastrigin_problem(dimensions=10):
    """
    Creates an instance of the Rastrigin problem.
    """

    def rastrigin_objective(x):
        """
        Objective function for the Rastrigin benchmark problem.
        """
        o_data = data_loader.load_mat_data('rastrigin_shift_func_data.mat')['o'][0]
        fbias_data = data_loader.load_mat_data('fbias_data.mat')['f_bias'][0]

        o = o_data[:dimensions]
        fbias = fbias_data[3]

        z = x - o

        obj = np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10) + fbias
        penalty = 0

        return obj, obj, penalty

    lower_bounds = [-5] * dimensions
    upper_bounds = [5] * dimensions

    return OptimizationProblem(
        name=f"rastrigin-{dimensions}D",
        objective_function=rastrigin_objective,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        is_minimization=True
    )
