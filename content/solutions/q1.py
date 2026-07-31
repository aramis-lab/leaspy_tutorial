# A logistic model with a single source, fitted like the 2-source one
model_1_source = LogisticModel(name="logistic", source_dimension=1)

model_1_source.fit(
    df_train, "mcmc_saem",
    seed=SEED, n_iter=1000, progress_bar=True,
    save_periodicity=500, plot_periodicity=500,
    overwrite_logs_folder=True,
)
