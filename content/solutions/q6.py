# Generate a new cohort from the fitted model and the visit design
simulated = model.simulate(
    algorithm="simulate",
    features=FEATURES,
    visit_parameters=visit_parameters,
    seed=SEED,
)

df_sim = simulated.data.to_dataframe().set_index(["ID", "TIME"])
