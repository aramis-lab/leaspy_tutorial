# Feature_1 ranges between 0 and 100 and is already increasing with disease severity. Just rescale it.
df_clean["Feature_1"] = df_clean["Feature_1"] / 100
