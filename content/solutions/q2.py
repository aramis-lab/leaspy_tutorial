# MoCA runs 0-31 and *decreases* with severity (31 = normal cognition).
# Rescale and flip in one step: 0 stays "normal", 1 becomes "most impaired".
df_clean["MOCA_total"] = 1 - df_clean["MOCA_total"] / 31

df_clean.agg(["min", "max"])
