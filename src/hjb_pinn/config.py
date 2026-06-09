from dataclasses import dataclass


@dataclass(frozen=True)
class SEIRConfig:
    birth_rate: float = 0.525
    death_rate: float = 0.5
    exposed_rate: float = 0.5
    recovery_rate: float = 0.1
    disease_death_rate: float = 0.2
    infection_cost: float = 0.3
    max_control: float = 0.9
    initial_state: tuple[float, float, float, float] = (
        1000.0,
        100.0,
        50.0,
        15.0,
    )


@dataclass(frozen=True)
class TrainingConfig:
    horizon: float = 20.0
    learning_rate: float = 5e-4
    adam_epochs: int = 10_000
    lbfgs_maxiter: int = 5_000
    collocation_size: int = 3_000
    terminal_size: int = 3_000
    state_dim: int = 4
    state_max: float = 2_000.0
    state_min_total: float = 100.0
    terminal_weight: float = 1.0
    value_scale: float = 40_000.0
    lbfgs_batches: int = 4
    log_every: int = 500


@dataclass(frozen=True)
class ExperimentConfig:
    conditioned: bool
    fixed_c: float = 0.002
    c_min: float = 0.001
    c_max: float = 0.003

    @property
    def c_range(self) -> float:
        return self.c_max - self.c_min
