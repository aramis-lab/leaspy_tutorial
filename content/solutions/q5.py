# Compare the two models: the lower the BIC / AIC, the better the fit
summary_2_sources = model_2_sources.summary()
summary_1_source = model_1_source.summary()

print("2 sources")
print(f"BIC: {float(summary_2_sources.bic)}")
print(f"AIC: {float(summary_2_sources.aic)}")
print("1 source")
print(f"BIC: {float(summary_1_source.bic)}")
print(f"AIC: {float(summary_1_source.aic)}")
