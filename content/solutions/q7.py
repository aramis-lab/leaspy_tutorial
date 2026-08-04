# Feature_2 is already between 0 and 1, but decreases with disease severity. Invert it.
df_clean["Feature_2"] = 1 - df_raw["Feature_2"]