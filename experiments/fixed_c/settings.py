from hjb_pinn import ExperimentConfig, SEIRConfig, TrainingConfig


SEIR = SEIRConfig()
TRAINING = TrainingConfig(log_every=500)
EXPERIMENT = ExperimentConfig(conditioned=False, fixed_c=0.002)
