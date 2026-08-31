# Both scores already increase with severity, so they only need rescaling.
# Divide each by the theoretical range of its questionnaire — not by the
# maximum observed here, which depends on who happens to be in the cohort.
df_clean["MDS1_total"] = df_clean["MDS1_total"] / 52    # MDS-UPDRS Part I: 0-52
df_clean["SCOPA_total"] = df_clean["SCOPA_total"] / 69  # SCOPA-AUT: 0-69

df_clean[["MDS1_total", "SCOPA_total"]].agg(["min", "max"])
