# Generate a new cohort from the fitted model and the visit design
simulated = model_100k.simulate(
    algorithm="simulate",
    features=FEATURES,
    visit_parameters=visit_parameters,
    seed=SEED,
)
