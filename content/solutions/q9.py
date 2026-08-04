features = FEATURES
colors = {"MDS1_total": "#1f77b4", "SCOPA_total": "#ff7f0e", "MOCA_total": "#2ca02c"}

fig, ax = plt.subplots(figsize=(14, 6))

for subject_id in df_train.index.get_level_values("ID").unique():
    subject_data = df_train.loc[subject_id]
    for feature in features:
        ax.plot(subject_data.index, subject_data[feature], alpha=0.8, color=colors[feature], linewidth=0.8)

for feature in features:
    ax.plot([], [], color=colors[feature], label=feature)

ax.set_xlabel("Age (years)")
ax.set_ylabel("Normalized score")
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.savefig('figures/spaghetti_plot_parkinson.pdf', dpi=800)
plt.show()