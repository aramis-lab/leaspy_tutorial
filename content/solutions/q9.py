# Compare the four models: the lower the BIC / AIC, the better the fit
models = {
    "Diagonal noise, 2 sources": model_2_sources,
    "Diagonal noise, 1 source": model_1_source,
    "Scalar noise, 2 sources": model_scalar_2_sources,
    "Scalar noise, 1 source": model_scalar_1_source,
}

for label, model in models.items():
    summary = model.summary()
    print(label)
    print(f"BIC: {float(summary.bic)}")
    print(f"AIC: {float(summary.aic)}")
