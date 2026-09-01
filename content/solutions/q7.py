# The same model and the same cohort, with two clusters instead of three.
model_mixture_2 = LogisticMultivariateMixtureModel(
    name="multi",
    source_dimension=2,
    dimension=6,
    n_clusters=2,
)

model_mixture_2.fit(
    data_mixture, "mcmc_saem", seed=1312, n_iter=100, progress_bar=True
)

# ICL = BIC + an entropy penalty on poorly separated clusters, so it reads in
# the same direction as the BIC and the AIC: the lower value wins.
icl_3 = float(model_mixture.summary().icl)
icl_2 = float(model_mixture_2.summary().icl)

print(f"3 clusters -> ICL {icl_3:.1f}")
print(f"2 clusters -> ICL {icl_2:.1f}")
print(f"Lower ICL: {3 if icl_3 < icl_2 else 2} clusters")

# Caution: at n_iter=100 neither chain has converged, so this ranking is not
# yet trustworthy - the point of the exercise is the comparison itself, not
# the winner. See the convergence section that follows.
