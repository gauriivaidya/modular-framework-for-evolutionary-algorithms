import numpy as np
import random

def eebs(population, minimize=True, **_):
    """
    Exploitation–Exploration Balance Selection (EEBS)
    Returns ONE selected individual (copy).

    NOTE: This function is stateful via function attributes, because your framework
    loads operators as plain callables without passing config params.
    """

    # ---- init persistent state on first call ----
    if not hasattr(eebs, "lambda_x"):
        eebs.lambda_x = 0.5
        eebs.lambda_y = 0.5
        eebs.min_lambda = 0.1
        eebs.step_mult_up = 1.1   
        eebs.step_mult_down = 0.9 
        eebs.rng = random.Random()

    if len(population) == 0:
        raise ValueError("Population is empty")

    K = len(population)
    fitnesses = np.array([ind.fitness for ind in population], dtype=float)

    # Rank BEST -> WORST (so exploitation pulls from best slice)
    order = np.argsort(fitnesses) if minimize else np.argsort(fitnesses)[::-1]
    ranked = [population[i] for i in order]

    cut = int(round(K * eebs.lambda_x))
    cut = max(0, min(K, cut))

    exploitation_list = ranked[:cut]   # best portion
    exploration_list = ranked[cut:]    # remainder

    # helper: keep lambdas sane + sum=1
    def clamp_norm():
        eebs.lambda_x = max(eebs.min_lambda, min(1.0, eebs.lambda_x))
        eebs.lambda_y = max(eebs.min_lambda, min(1.0, eebs.lambda_y))
        s = eebs.lambda_x + eebs.lambda_y
        eebs.lambda_x /= s
        eebs.lambda_y /= s

    # ---- pseudocode logic ----
    if eebs.lambda_x < eebs.lambda_y:
        eebs.lambda_x = min(1.0, eebs.lambda_x * eebs.step_mult_up)
        eebs.lambda_y = max(eebs.min_lambda, eebs.lambda_y * eebs.step_mult_down)
        clamp_norm()

        pool = exploration_list if exploration_list else ranked
        return eebs.rng.choice(pool).copy()

    if eebs.lambda_x > eebs.lambda_y:
        eebs.lambda_x = max(eebs.min_lambda, eebs.lambda_x * eebs.step_mult_down)
        eebs.lambda_y = min(1.0, eebs.lambda_y * eebs.step_mult_up)
        clamp_norm()

        pool = exploitation_list if exploitation_list else ranked
        return eebs.rng.choice(pool).copy()

    # equal
    return eebs.rng.choice(ranked).copy()