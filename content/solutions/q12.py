# Tell Leaspy that this dataframe contains both visits and an event outcome.
data_joint = Data.from_dataframe(df_joint, data_type="joint")

print(f"{data_joint.n_individuals} subjects")
print(f"{data_joint.n_visits} visits")
print(f"Longitudinal features: {data_joint.headers}")
