from pathlib import Path

import equinox as eqx
import jax.random as jr


ROOT = Path(__file__).resolve().parents[2]

from hjb_pinn.model import make_model
from hjb_pinn.plotting import plot_simulation
from hjb_pinn.simulation import simulate

from experiments.fixed_c.settings import EXPERIMENT, SEIR, TRAINING


OUTPUT_DIR = ROOT / "outputs" / "fixed_c"


def main():
    model_path = OUTPUT_DIR / "model.eqx"
    skeleton = make_model(jr.PRNGKey(42), TRAINING, EXPERIMENT)
    model = eqx.tree_deserialise_leaves(model_path, skeleton)
    times, controlled, uncontrolled, controls = simulate(
        model, EXPERIMENT.fixed_c, SEIR, TRAINING
    )
    figure_path = OUTPUT_DIR / "seir_states.png"
    plot_simulation(
        times,
        controlled,
        uncontrolled,
        controls,
        figure_path,
        title="SEIR Optimal Control via HJB-PINN",
        max_control=SEIR.max_control,
    )
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
