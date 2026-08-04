# —— Let's select patients with at least two visits
indices = [idx for idx in df_clean.index.unique("ID") if df_clean.loc[idx].shape[0] >= 2]
df_clean = df_clean[df_clean.index.get_level_values(0).isin(indices)]
df_clean.head()