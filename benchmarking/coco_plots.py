
from pathlib import Path

import matplotlib.pyplot as plt


def plot_convergence(
    history,
    output_path,
    algorithm_name="Optimizer",
    problem_name="COCO"
):
    if not history:
        print("No convergence history available.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    use_evaluations = all(
        "evaluations" in item for item in history
    )

    x_key = (
        "evaluations"
        if use_evaluations
        else "generation"
    )

    x = [item[x_key] for item in history]
    y = [item["fitness"] for item in history]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        x,
        y,
        linewidth=2,
        label=algorithm_name
    )

    ax.set_xlabel(
        "Objective evaluations"
        if use_evaluations
        else "Generation"
    )

    ax.set_ylabel("Best Fitness")

    ax.set_title(
        f"{algorithm_name} Convergence on {problem_name}"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Convergence graph saved: {output_path}")
    