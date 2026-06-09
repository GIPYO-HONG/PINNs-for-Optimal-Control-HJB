from hjb_pinn import ExperimentConfig, SEIRConfig, TrainingConfig


SEIR = SEIRConfig()
TRAINING = TrainingConfig(log_every=1_000)
EXPERIMENT = ExperimentConfig(
    conditioned=True,
    c_min=0.001,
    c_max=0.003,
)
