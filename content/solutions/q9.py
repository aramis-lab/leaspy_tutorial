# Personalize the 100k-iteration model on the partial test set (last visit removed)
ip_test = model_100k.personalize(data_test_partial, "scipy_minimize", seed=SEED, progress_bar=True)
