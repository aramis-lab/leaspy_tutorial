# Datasets

| file | what it is |
|---|---|
| `parkinson.csv` | Leaspy's synthetic Parkinson's dataset (v2.1.0), **normalized** — this is the one the notebooks fit. 1,997 visits, 200 patients. |
| `parkinson_original_scale.csv` | The same data put back on the **original clinical scales**. Same rows, same `ID` / `TIME`, for reading and plotting — not for fitting. |
| `simulated_data_for_joint_model.csv` | Simulated ALS data for the joint-model section: 150 subjects, ALSFRS-R subscores plus `EVENT_TIME` / `EVENT_BOOL`. |

## Why `parkinson.csv` doesn't look like clinical data

Leaspy assumes every marker is **normalized to `[0, 1]`** and **increases with severity**. The
dataset ships already prepared that way, which is convenient for fitting but makes the numbers
unreadable: an MDS-UPDRS I of `0.112` means nothing to a clinician.

That preparation is not something invented for this tutorial. It comes from the study the dataset
mimics (Couronné et al., *Charting Parkinson's disease progression*; see also Couronné's thesis
§4.3.2, "Data normalization"), whose rule is:

> **0 = normal value, 1 = maximum pathological change.**
> Markers that decrease with progression were reversed. Clinical endpoints were scaled by the
> theoretical range of the instrument. Imaging endpoints were anchored at 0 = mean striatal
> binding ratio (SBR) of controls at baseline, 1 = a null SBR.

`parkinson_original_scale.csv` simply inverts that, feature by feature.

## The transformation, feature by feature

| feature | instrument | scale used | reversed? | back to original scale |
|---|---|---|---|---|
| `MDS1_total` | MDS-UPDRS Part I | 0–52 | no | `x × 52` |
| `MDS2_total` | MDS-UPDRS Part II | 0–52 | no | `x × 52` |
| `MDS3_off_total` | MDS-UPDRS Part III (OFF) | 0–108 | no | `x × 108` |
| `SCOPA_total` | SCOPA-AUT | 0–69 | no | `x × 69` |
| `MOCA_total` | MoCA | 0–31 | **yes** | `31 × (1 − x)` |
| `REM_total` | RBDSQ | 0–13 | no | `x × 13` |
| `PUTAMEN_R` / `PUTAMEN_L` | Putamen SBR (DaTscan) | 0 – 2.1 | **yes** | `2.1 × (1 − x)` |
| `CAUDATE_R` / `CAUDATE_L` | Caudate SBR (DaTscan) | 0 – 3.0 | **yes** | `3.0 × (1 − x)` |

**Why those directions.** MDS-UPDRS, SCOPA-AUT and RBDSQ are *symptom burden* scores — a higher
number already means a sicker patient, so nothing had to be flipped. MoCA is a *cognition* score
(31 = normal, 0 = maximal impairment) and the SBRs measure *surviving dopaminergic signal*, which
falls as the disease advances. Those five are the ones that were reversed, and reversing them back
is what `1 − x` does.

**Why those bounds.** For the clinical scores, the bound is the range of the questionnaire itself.
For the imaging markers there is no theoretical maximum, so the study anchored the "normal" end at
the mean SBR of PPMI healthy controls at baseline — 2.1 for putamen, 3.0 for caudate — and the
pathological end at zero signal. The control mean is the same on both sides, so the same constant
applies to `_R` and `_L` (the study reports SBR by ipsi-/contralateral hemisphere rather than
right/left; this does not change the arithmetic).

Two bounds deviate from the textbook value, and the study's numbers are used here because they
are what produced the file:

- **MDS-UPDRS III → 108, not 132.** The instrument's theoretical maximum is 132 (33 items × 4),
  but the normalization used 108. Using 132 would inflate every Part III value by ~22%.
- **MoCA → 31, not 30.** The standard MoCA caps at 30; the study states 0–31 (the
  education-adjustment point).

## How faithful is it?

Re-applying the normalization to `parkinson_original_scale.csv` reproduces `parkinson.csv` to
within `1.1e-16`, so nothing is lost — the two files hold the same information in different units.

The reconstructed baseline values also land on the published PPMI cohort statistics, which is the
real check that the constants are the right ones:

| marker | reconstructed baseline | published PPMI PD baseline |
|---|---|---|
| MDS-UPDRS I | 6.1 ± 4.3 | 5.8 ± 4.2 |
| MDS-UPDRS II | 6.5 ± 4.0 | 5.7 ± 4.2 |
| MDS-UPDRS III Off | 22.4 ± 9.1 | 22.8 ± 10.3 |
| SCOPA-AUT | 12.8 ± 7.9 | 9.5 ± 6.1 |
| MoCA | 28.2 ± 2.5 | 27.1 ± 2.3 |
| RBDSQ | 3.6 ± 2.7 | 4.1 ± 2.7 |
| Putamen SBR (R / L) | 0.8 / 0.7 | 0.7 (contra) / 1.0 (ipsi) |
| Caudate SBR (R / L) | 1.9 / 1.9 | 1.8 (contra) / 2.2 (ipsi) |

## Two caveats

- **Values are not integers.** `load_dataset("parkinson")` is *synthetic* data — Leaspy describes
  it as "synthetic longitudinal observations mimicking cohort of subjects with neurodegenerative
  disorders". No integer grid underlies it (no multiplier from 1 to 300 makes the normalized values
  integral), so the inverse yields an MDS-UPDRS I of `5.84`, not `6`. Rounding would look more like
  a real case report form but would not recover anything real.
- **Clipped extremes are preserved.** An exact `0.0` in the normalized file becomes the normal end
  of the raw scale (MoCA 31, `CAUDATE_R` 3.00), and `REM_total` sits at 13/13 for 8 visits.
