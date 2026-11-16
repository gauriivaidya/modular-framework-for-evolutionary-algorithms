import numpy as np
from core.problem import OptimizationProblem


def spring_design_objective(x, penalty_value=1e5):
    """
    Objective function for the Spring Design problem.
    It returns the objective value, the clean objective value, and the penalty.
    """
    # Objective function
    obj = (x[2] + 2) * x[1] * (x[0] ** 2)

    # Constraints
    g1 = 1 - ((x[1] ** 3) * x[2]) / (71785 * (x[0] ** 4))
    g2 = (4 * (x[1] ** 2) - x[0] * x[1]) / (12566 * (x[1] * (x[0] ** 3) - x[0] ** 4)) + (1 / (5108 * (x[0] ** 2))) - 1
    g3 = 1 - (140.45 * x[0]) / ((x[1] ** 2) * x[2])
    g4 = (x[0] + x[1]) / 1.5 - 1

    # Penalty calculation
    penalty = 0
    constraints = [g1, g2, g3, g4]
    for g in constraints:
        if g > 0:
            penalty += penalty_value * g

    return obj + penalty, obj, penalty


# Define the bounds for the Spring Design problem
lower_bounds = [0.05, 0.25, 2.0]
upper_bounds = [2.0, 1.3, 15.0]

# Create an instance of the OptimizationProblem
spring_design_problem = OptimizationProblem(
    name="spring_design",
    objective_function=spring_design_objective,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    is_minimization=True
)
