# Solution for exercise 3 — build a small longitudinal dataframe `df`.
#
# This is what tutolib.solution(3) reveals and (on "Run it for me") executes in
# the notebook, so `df` exists afterwards even if the user got stuck.
# Self-contained on purpose (no external file) so the template runs anywhere.
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
rows = []
for pid in range(1, 6):                         # 5 patients
    onset = rng.uniform(60, 75)                 # individual time-shift (tau-like)
    for visit in range(4):                      # 4 visits each
        age = onset + 1.5 * visit
        score = 1 / (1 + np.exp(-(age - 68)))   # logistic progression
        rows.append((f"P{pid:03d}", round(age, 1), round(score + rng.normal(0, 0.03), 3)))

df = pd.DataFrame(rows, columns=["ID", "TIME", "SCORE"]).set_index(["ID", "TIME"])
df
