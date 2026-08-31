# —— Let's select patients with at least two visits
n_visits = df_clean.groupby("ID").size()
kept = n_visits[n_visits >= 2].index
df_clean = df_clean[df_clean.index.get_level_values("ID").isin(kept)]

print(f"{n_visits.size - kept.size} subject(s) dropped, {kept.size} kept.")
df_clean.head()
