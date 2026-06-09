from pathlib import Path

import matplotlib.pyplot as plt


STATE_NAMES = ("S", "E", "I", "R")
STATE_COLORS = ("#1E88E5", "#FB8C00", "#E53935", "#43A047")


def plot_simulation(
    times,
    controlled,
    uncontrolled,
    controls,
    output_path: Path,
    title: str,
    max_control: float,
    color: str | None = None,
):
    figure, axes = plt.subplots(2, 3, figsize=(21, 10))
    figure.suptitle(title, fontsize=16, fontweight="bold")

    for index, (name, default_color) in enumerate(
        zip(STATE_NAMES, STATE_COLORS)
    ):
        axis = axes.flat[index]
        line_color = color or default_color
        axis.plot(
            times,
            controlled[:, index],
            color=line_color,
            linewidth=2.5,
            label=f"{name} Controlled",
        )
        axis.plot(
            times,
            uncontrolled[:, index],
            color=line_color,
            linestyle="--",
            linewidth=2.0,
            alpha=0.6,
            label=f"{name} Uncontrolled",
        )
        axis.fill_between(
            times,
            controlled[:, index],
            uncontrolled[:, index],
            color=line_color,
            alpha=0.12,
        )
        axis.set_title(f"{name}(t)")
        axis.set_ylabel(name)

    total_axis = axes.flat[4]
    total_color = color or "#6D4C41"
    total_axis.plot(
        times,
        controlled.sum(axis=1),
        color=total_color,
        linewidth=2.5,
        label="N Controlled",
    )
    total_axis.plot(
        times,
        uncontrolled.sum(axis=1),
        color=total_color,
        linestyle="--",
        linewidth=2.0,
        alpha=0.6,
        label="N Uncontrolled",
    )
    total_axis.set_title("N(t)")
    total_axis.set_ylabel("N")

    control_axis = axes.flat[5]
    control_color = color or "#8E24AA"
    control_axis.plot(
        times,
        controls,
        color=control_color,
        linewidth=2.5,
        label="Optimal Control",
    )
    control_axis.fill_between(
        times, 0.0, controls, color=control_color, alpha=0.15
    )
    control_axis.axhline(max_control, color="gray", linestyle=":")
    control_axis.set_ylim(-0.05, max_control + 0.1)
    control_axis.set_title("Optimal Vaccination Control u(t)")
    control_axis.set_ylabel("u(t)")

    for axis in axes.flat:
        axis.set_xlabel("Time")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=9, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
