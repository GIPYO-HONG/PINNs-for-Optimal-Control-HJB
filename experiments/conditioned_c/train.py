from pathlib import Path

import equinox as eqx
import jax.random as jr
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

from hjb_pinn.model import make_model
from hjb_pinn.training import train

from experiments.conditioned_c.settings import EXPERIMENT, SEIR, TRAINING


OUTPUT_DIR = ROOT / "outputs" / "conditioned_c"


def main():
    key = jr.PRNGKey(42)
    model = make_model(key, TRAINING, EXPERIMENT)
    model, history = train(model, key, SEIR, TRAINING, EXPERIMENT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "model_c.eqx"
    history_path = OUTPUT_DIR / "loss_history.npz"
    eqx.tree_serialise_leaves(model_path, model)
    np.savez(
        history_path,
        pde=np.asarray(history["pde"]),
        terminal=np.asarray(history["terminal"]),
        total=np.asarray(history["total"]),
    )
    print(f"\nSaved:\n - {model_path}\n - {history_path}")


if __name__ == "__main__":
    main()
