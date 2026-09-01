# Same 1-source model, but with a single noise std shared by all the scores
model_scalar_1_source = LogisticModel(
    name="logistic", source_dimension=1, obs_models="gaussian-scalar"
)

model_scalar_1_source.fit(
    df_train, "mcmc_saem",
    seed=SEED, n_iter=1000, progress_bar=True,
    save_periodicity=500,
    path="_outputs/model_scalar_1_source",
    overwrite_logs_folder=True,
)
