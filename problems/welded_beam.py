import numpy as np
from core.problem import OptimizationProblem


def welded_beam_objective(x, penalty_value=1e5):
    """
    Objective function for the Welded Beam problem.
    It returns the objective value, the clean objective value, and the penalty.
    """
    # Objective function
    obj = 1.10471 * (x[0] ** 2) * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])

    # Constraints
    P = 6000
    L = 14
    E = 30e6
    G = 12e6
    t_max = 13600
    s_max = 30000
    d_max = 0.25

    M = P * (L + x[1] / 2.0)
    R = np.sqrt((x[1] ** 2) / 4.0 + ((x[0] + x[2]) / 2.0) ** 2)
    J = 2 * (np.sqrt(2) * x[0] * x[1] * (((x[1] ** 2) / 12.0) + ((x[0] + x[2]) / 2.0) ** 2))

    t1 = P / (np.sqrt(2) * x[0] * x[1])
    t2 = M * R / J
    t = np.sqrt(t1 ** 2 + 2 * t1 * t2 * x[1] / (2 * R) + t2 ** 2)
    s = 6 * P * L / (x[3] * x[2] ** 2)
    d = 4 * P * (L ** 3) / (E * x[3] * x[2] ** 3)

    Pc_term1 = 4.013 * E * np.sqrt((x[2] ** 2 * x[3] ** 6) / 36.0) / (L ** 2)
    Pc_term2 = 1 - (x[2] / (2 * L)) * np.sqrt(E / (4 * G))
    Pc = Pc_term1 * Pc_term2

    # Constraint violations
    g1 = t - t_max
    g2 = s - s_max
    g3 = x[0] - x[3]
    g4 = 0.10471 * (x[0] ** 2) + 0.04811 * x[2] * x[3] * (14.0 + x[1]) - 5.0
    g5 = 0.125 - x[0]
    g6 = d - d_max
    g7 = P - Pc

    # Penalty calculation
    penalty = 0
    constraints = [g1, g2, g3, g4, g5, g6, g7]
    for g in constraints:
        if g > 0:
            penalty += penalty_value * g

    return obj + penalty, obj, penalty


# Define the bounds for the Welded Beam problem
lower_bounds = [0.1, 0.1, 0.1, 0.1]
upper_bounds = [2, 10, 10, 2]

welded_beam_problem = OptimizationProblem(
    name="welded_beam",
    objective_function=welded_beam_objective,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    is_minimization=True
)
