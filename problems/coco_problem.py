import numpy as np
import cocoex


class COCOProblem:
    def __init__(
        self,
        suite="bbob",
        function=1,
        dimension=5,
        instance=1
    ):
        self.suite = cocoex.Suite(
            suite,
            "",
            f"dimensions:{dimension} "
            f"function_indices:{function} "
            f"instance_indices:{instance}"
        )

        self.problem = self.suite.get_problem(0)

        self.name = self.problem.id
        self.dimension = self.problem.dimension
        self.size = self.dimension
        self.num_dimensions = self.dimension
        self.is_minimization = True

        self.lower_bounds = np.asarray(
            self.problem.lower_bounds,
            dtype=float
        )

        self.upper_bounds = np.asarray(
            self.problem.upper_bounds,
            dtype=float
        )

    def evaluate(self, solution):
        x = np.asarray(solution, dtype=float)

        if x.shape != (self.dimension,):
            raise ValueError(
                f"Expected shape ({self.dimension},), "
                f"got {x.shape}"
            )

        if not np.all(np.isfinite(x)):
            return float("inf")

        return float(self.problem(x))

    def close(self):
        self.problem.free()
        self.suite.free()


def create_coco_problem(
    suite="bbob",
    function=1,
    dimension=5,
    instance=1
):
    return COCOProblem(
        suite=suite,
        function=function,
        dimension=dimension,
        instance=instance
    )