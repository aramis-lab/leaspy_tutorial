# And the 2-source model with a scalar noise
model_scalar_2_sources = LogisticModel(
    name="logistic", source_dimension=2, obs_models="gaussian-scalar"
)

model_scalar_2_sources.fit(
    df_train, "mcmc_saem",
    seed=SEED, n_iter=1000, progress_bar=True,
    save_periodicity=500,
    overwrite_logs_folder=True,
)
