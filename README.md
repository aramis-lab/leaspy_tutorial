# Leaspy tutorial

Hands-on tutorial for **[Leaspy](https://gitlab.com/icm-institute/aramislab/leaspy) v2.1**,
for the ICM / Paris Brain Institute workshop
*[Challenges from Longitudinal Data](https://parisbraininstitute.org/agenda/challenges-longitudinal-data)*
(Sept 3-4, 2026). The notebooks run on **Google Colab** — no local install needed.

## For participants

Open a notebook in Colab and run the first cell. It pins the right versions and
loads the helper library, so everyone is on the same setup:

```python
!pip install -q "leaspy==2.1.*"
!git clone -q --branch v1.0 https://github.com/aramis-lab/leaspy_tutorial
import sys; sys.path.insert(0, "leaspy_tutorial")
import tutolib as tp
```

Throughout the notebooks you'll meet two helpers:

- `tp.runquestion(<id>)` — a clickable multiple-choice question that tells you if
  you're right and gives feedback if not.
- `tp.solution(<id>)` — for "fill-in" cells: reveals the solution and can **run it
  for you**, so you're never blocked from continuing.

See [`notebooks/00_template.ipynb`](notebooks/00_template.ipynb) for a 1-minute demo of both.

## Repo layout

```
tutolib/                 the mini helper library (runquestion / solution)
content/
  questions.yaml         multiple-choice questions (edit this to add questions)
  solutions/q<id>.py     one solution snippet per fill-in exercise
notebooks/               the tutorial notebooks (00_template is the demo)
data/                    datasets used by the tutorial
```

## For contributors

We author the **challenge** notebook directly (cells the user fills are left as
`# To complete`) and keep each answer as a separate file the helper reads at
runtime — no notebook-generation step.

**Adding a multiple-choice question:** add a block to
[`content/questions.yaml`](content/questions.yaml) with a new id, then call
`tp.runquestion(<id>)` in a cell. No Python needed.

**Adding a fill-in exercise:** leave a `# To complete` cell for the user, write the
answer in `content/solutions/q<id>.py`, and add a `tp.solution(<id>)` cell after it.

**Workflow:** each of us works on their own branch. Before the workshop we run every
notebook top-to-bottom once against the pinned Leaspy version, then cut the `v1.0`
git tag — that frozen tag is what the setup cell clones.
