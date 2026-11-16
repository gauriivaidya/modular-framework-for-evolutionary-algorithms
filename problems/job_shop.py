from pathlib import Path
import re
import numpy as np


class JobShopProblem:

    def __init__(
        self,
        jobs,
        num_machines,
        name="job_shop"
    ):
        """
        jobs[job_id] = [
            (machine_id, processing_time),
            ...
        ]
        """

        self.jobs = [
            [
                (int(machine), int(duration))
                for machine, duration in job
            ]
            for job in jobs
        ]

        self.num_jobs = len(self.jobs)
        self.num_machines = int(num_machines)

        self.operations_per_job = [
            len(job)
            for job in self.jobs
        ]

        self.num_operations = sum(
            self.operations_per_job
        )

        self.size = self.num_operations
        self.dimension = self.num_operations
        self.num_dimensions = self.num_operations

        self.name = name
        self.is_minimization = True

    # =====================================================
    # Schedule decoder
    # =====================================================

    def _decode(self, solution):
        """
        Decode an operation-based chromosome.

        Each value is a job id.

        Example:
            [0, 1, 0, 2, 1, ...]

        The first occurrence of job 0 schedules operation 0
        of job 0, the second occurrence schedules operation 1,
        etc.

        Returns:
            makespan,
            schedule
        """

        solution = list(solution)

        next_operation = [
            0
            for _ in range(self.num_jobs)
        ]

        job_ready = [
            0
            for _ in range(self.num_jobs)
        ]

        machine_ready = [
            0
            for _ in range(self.num_machines)
        ]

        schedule = []

        for job_id in solution:

            job_id = int(job_id)

            if not (
                0 <= job_id < self.num_jobs
            ):
                raise ValueError(
                    f"Invalid job id: {job_id}"
                )

            operation_index = (
                next_operation[job_id]
            )

            if (
                operation_index
                >= len(self.jobs[job_id])
            ):
                raise ValueError(
                    f"Job {job_id} occurs too many times "
                    f"in the solution."
                )

            machine, duration = (
                self.jobs[job_id][operation_index]
            )

            start = max(
                job_ready[job_id],
                machine_ready[machine]
            )

            finish = start + duration

            schedule.append({
                "job": job_id,
                "operation": operation_index,
                "machine": machine,
                "start": start,
                "finish": finish
            })

            job_ready[job_id] = finish
            machine_ready[machine] = finish

            next_operation[job_id] += 1

        makespan = (
            max(job_ready)
            if job_ready
            else 0
        )

        return (
            float(makespan),
            schedule
        )

    # =====================================================
    # Objective
    # =====================================================

    def evaluate(self, solution):
        """
        Objective:
            minimize makespan.
        """

        solution = list(solution)

        if len(solution) != self.num_operations:
            raise ValueError(
                f"Expected {self.num_operations} operations, "
                f"got {len(solution)}."
            )

        counts = [
            0
            for _ in range(self.num_jobs)
        ]

        for job_id in solution:
            counts[int(job_id)] += 1

        for job_id in range(
            self.num_jobs
        ):
            if (
                counts[job_id]
                != self.operations_per_job[job_id]
            ):
                raise ValueError(
                    f"Job {job_id} should occur "
                    f"{self.operations_per_job[job_id]} times, "
                    f"but occurs {counts[job_id]} times."
                )

        makespan, _ = self._decode(
            solution
        )

        return makespan

    # =====================================================
    # ACO state
    # =====================================================

    def initial_state(self):
        """
        State is a partial sequence of job ids.
        """
        return []

    def feasible_moves(self, state):
        """
        A job is feasible if it still has
        unscheduled operations.
        """

        if len(state) >= self.num_operations:
            return []

        counts = [
            0
            for _ in range(self.num_jobs)
        ]

        for job_id in state:
            counts[int(job_id)] += 1

        return [
            job_id
            for job_id in range(self.num_jobs)
            if (
                counts[job_id]
                < self.operations_per_job[job_id]
            )
        ]

    def apply_move(
        self,
        state,
        move
    ):
        return state + [int(move)]

    def is_complete(
        self,
        state
    ):
        return (
            len(state)
            == self.num_operations
        )

    def finalize_solution(
        self,
        state
    ):
        return list(state)

    # =====================================================
    # Pheromone
    # =====================================================

    def pheromone_key(
        self,
        state,
        move
    ):
        """
        Pheromone represents:
            construction position -> chosen job
        """

        position = len(state)

        return (
            position,
            int(move)
        )

    def all_components(self):
        return [
            (position, job_id)
            for position
            in range(self.num_operations)
            for job_id
            in range(self.num_jobs)
        ]

    def solution_components(
        self,
        solution
    ):
        return [
            (
                position,
                int(job_id)
            )
            for position, job_id
            in enumerate(solution)
        ]

    # =====================================================
    # Heuristic
    # =====================================================

    def heuristic_value(
        self,
        state,
        move
    ):
        """
        Prefer jobs whose next operation
        has shorter processing time.
        """

        job_id = int(move)

        occurrence = sum(
            1
            for x in state
            if int(x) == job_id
        )

        if (
            occurrence
            >= len(self.jobs[job_id])
        ):
            return 1e-12

        _, duration = (
            self.jobs[job_id][occurrence]
        )

        return 1.0 / max(
            float(duration),
            1e-12
        )

    # =====================================================
    # Helpers
    # =====================================================

    def solution_to_array(
        self,
        solution
    ):
        return np.asarray(
            solution,
            dtype=int
        )

    def neighbors(
        self,
        solution
    ):
        """
        Swap two positions.

        Since the chromosome contains the same
        multiset of job IDs, swapping preserves
        the number of operations for each job.
        """

        solution = list(solution)

        n = len(solution)

        for i in range(n):
            for j in range(
                i + 1,
                n
            ):

                if solution[i] == solution[j]:
                    continue

                neighbor = solution[:]

                neighbor[i], neighbor[j] = (
                    neighbor[j],
                    neighbor[i]
                )

                yield neighbor


# =========================================================
# OR-Library loader
# =========================================================

def load_job_shop_instance(
    filepath,
    instance_name
):
    """
    Load one named instance from OR-Library jobshop1.txt.

    Examples:
        abz5
        ft06
        la01
        orb01
        swv01
        yn1
    """

    filepath = Path(filepath)

    text = filepath.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    target = instance_name.strip().lower()

    instance_line = None

    pattern = re.compile(
        r"^\s*instance\s+(\S+)",
        re.IGNORECASE
    )

    for index, line in enumerate(
        lines
    ):
        match = pattern.match(line)

        if (
            match
            and match.group(1).lower()
            == target
        ):
            instance_line = index
            break

    if instance_line is None:
        raise ValueError(
            f"Instance '{instance_name}' "
            f"not found in {filepath.name}."
        )

    # -----------------------------------------------------
    # Find the dimensions line after the instance marker.
    #
    # Description text and separator lines occur between
    # "instance la01" and "10 5".
    # -----------------------------------------------------

    dimensions_index = None
    num_jobs = None
    num_machines = None

    for index in range(
        instance_line + 1,
        len(lines)
    ):

        stripped = lines[index].strip()

        # Reached next instance unexpectedly
        if (
            index > instance_line + 1
            and pattern.match(lines[index])
        ):
            break

        values = stripped.split()

        if (
            len(values) == 2
            and all(
                value.lstrip("-").isdigit()
                for value in values
            )
        ):
            num_jobs = int(values[0])
            num_machines = int(values[1])

            dimensions_index = index
            break

    if dimensions_index is None:
        raise ValueError(
            f"Could not find dimensions for "
            f"instance '{instance_name}'."
        )

    # -----------------------------------------------------
    # Read one row per job.
    #
    # Each row contains:
    # machine processing_time
    # repeated num_machines times.
    # -----------------------------------------------------

    jobs = []

    line_index = (
        dimensions_index + 1
    )

    for job_id in range(
        num_jobs
    ):

        while (
            line_index < len(lines)
            and not lines[line_index].strip()
        ):
            line_index += 1

        if line_index >= len(lines):
            raise ValueError(
                f"Unexpected end of file while "
                f"reading job {job_id}."
            )

        values = [
            int(x)
            for x
            in lines[line_index].split()
        ]

        line_index += 1

        expected_values = (
            2 * num_machines
        )

        if len(values) != expected_values:
            raise ValueError(
                f"Job {job_id} in '{instance_name}' "
                f"has {len(values)} values; expected "
                f"{expected_values}."
            )

        job = []

        for operation in range(
            num_machines
        ):

            machine = values[
                2 * operation
            ]

            duration = values[
                2 * operation + 1
            ]

            if not (
                0
                <= machine
                < num_machines
            ):
                raise ValueError(
                    f"Invalid machine {machine} "
                    f"in job {job_id}."
                )

            job.append(
                (
                    machine,
                    duration
                )
            )

        jobs.append(
            job
        )

    return (
        jobs,
        num_machines
    )


# =========================================================
# Factory
# =========================================================

def create_job_shop_problem(
    instance="jobshop1.txt",
    test_problem="ft06"
):

    root = (
        Path(__file__).resolve()
        .parent.parent
        / "data"
        / "or_library"
        / "job_shop"
    )

    filepath = root / instance

    if not filepath.exists():

        raise FileNotFoundError(
            f"Job Shop instance file "
            f"not found: {filepath}"
        )

    (
        jobs,
        num_machines
    ) = load_job_shop_instance(
        filepath,
        instance_name=test_problem
    )

    return JobShopProblem(
        jobs=jobs,
        num_machines=num_machines,
        name=test_problem
    )