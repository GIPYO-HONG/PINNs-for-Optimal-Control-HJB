from pathlib import Path

import equinox as eqx
import jax.random as jr


ROOT = Path(__file__).resolve().parents[2]

from hjb_pinn.model import make_model
from hjb_pinn.plotting import plot_simulation
from hjb_pinn.simulation import simulate

from experiments.conditioned_c.settings import EXPERIMENT, SEIR, TRAINING


OUTPUT_DIR = ROOT / "outputs" / "conditioned_c"
C_VALUES = (0.001, 0.002, 0.003)
COLORS = ("#1E88E5", "#FB8C00", "#E53935")


def main():
    model_path = OUTPUT_DIR / "model_c.eqx"
    skeleton = make_model(jr.PRNGKey(42), TRAINING, EXPERIMENT)
    model = eqx.tree_deserialise_leaves(model_path, skeleton)

    for c_val, color in zip(C_VALUES, COLORS):
        times, controlled, uncontrolled, controls = simulate(
            model, c_val, SEIR, TRAINING
        )
        suffix = str(c_val).replace(".", "")
        figure_path = OUTPUT_DIR / f"seir_c{suffix}.png"
        plot_simulation(
            times,
            controlled,
            uncontrolled,
            controls,
            figure_path,
            title=f"SEIR Dynamics: Optimal Control vs No Control (c={c_val})",
            max_control=SEIR.max_control,
            color=color,
        )
        print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
