"""tutolib — tiny helpers for the Leaspy tutorial notebooks.

Call these from a notebook cell::

    import tutolib as tp
    tp.runquestion(1)     # interactive multiple-choice question
    tp.solution(3)        # reveal a solution snippet, and run it for you

Content is kept *outside* the code so anyone can author it without touching
Python:

    content/questions.yaml         multiple-choice questions, keyed by id
    content/solutions/q<id>.py     one solution snippet per fill-in exercise
"""
from __future__ import annotations

from pathlib import Path

from IPython.display import Code, display

try:
    import ipywidgets as widgets
except Exception:  # always present in Colab; keep `import tutolib` safe elsewhere
    widgets = None

__version__ = "0.1.0"

# content/ sits at the repo root, next to tutolib/ — resolve it from this file so
# it works whatever the notebook's working directory is.
_CONTENT = Path(__file__).resolve().parent.parent / "content"
_QUESTIONS_FILE = _CONTENT / "questions.yaml"
_SOLUTIONS_DIR = _CONTENT / "solutions"

_questions_cache = None


def _load_questions():
    global _questions_cache
    if _questions_cache is None:
        import yaml  # imported lazily: only needed once a question is shown
        with open(_QUESTIONS_FILE, encoding="utf-8") as f:
            _questions_cache = yaml.safe_load(f) or {}
    return _questions_cache


def _ipython():
    try:
        from IPython import get_ipython
        return get_ipython()
    except Exception:
        return None


def runquestion(qid):
    """Render multiple-choice question ``qid`` as a clickable widget.

    Reads the question from ``content/questions.yaml`` (prompt / options /
    answer / feedback) and shows radio buttons + a *Check* button. On check it
    says whether the pick is right and, if wrong, displays the feedback.
    """
    q = _load_questions().get(qid)
    if q is None:
        raise KeyError(f"No question {qid!r} in {_QUESTIONS_FILE}")

    options = list(q["options"])
    answer = q["answer"]                 # 0-based index of the correct option
    feedback = q.get("feedback", "")

    if widgets is None:                  # plain-text fallback (no ipywidgets)
        print(q["prompt"])
        for i, opt in enumerate(options):
            print(f"  [{i}] {opt}")
        return

    title = widgets.HTML(f"<b>❓ {q['prompt']}</b>")
    choices = widgets.RadioButtons(
        options=[(opt, i) for i, opt in enumerate(options)],
        index=None,                      # nothing selected at first
        layout=widgets.Layout(width="auto"),
    )
    check = widgets.Button(description="Check", button_style="primary")
    out = widgets.Output()

    def _on_check(_):
        with out:
            out.clear_output()
            if choices.index is None:
                print("⚠️  Pick an answer first.")
            elif choices.value == answer:
                display(widgets.HTML("<span style='color:#1e7e34'>✅ Correct!</span>"))
            else:
                html = "<span style='color:#c0392b'>❌ Not quite.</span>"
                if feedback:
                    html += f"<br>{feedback}"
                display(widgets.HTML(html))

    check.on_click(_on_check)
    display(widgets.VBox([title, choices, check, out]))


def solution(qid, run=False):
    """Reveal the solution snippet for fill-in exercise ``qid``.

    Reads ``content/solutions/q<qid>.py`` and shows a *Show solution* button.
    Once revealed, a *Run it for me* button executes the snippet in the
    notebook's own namespace via ``get_ipython().run_cell`` — so the variables
    the rest of the notebook needs get defined even if the user is stuck or
    skips the exercise. Pass ``run=True`` to reveal and run immediately.
    """
    path = _SOLUTIONS_DIR / f"q{qid}.py"
    if not path.exists():
        raise FileNotFoundError(f"No solution file at {path}")
    code = path.read_text(encoding="utf-8")

    def _run():
        ip = _ipython()
        if ip is not None:
            ip.run_cell(code)            # runs as if typed: defines vars, draws plots
        else:
            exec(compile(code, str(path), "exec"), {})

    if run or widgets is None:
        display(Code(code, language="python"))
        _run()
        return

    show = widgets.Button(description="Show solution", icon="eye")
    box = widgets.Output()

    def _on_show(_):
        with box:
            box.clear_output()
            display(Code(code, language="python"))
            run_btn = widgets.Button(
                description="Run it for me", button_style="success", icon="play"
            )
            run_btn.on_click(lambda _: _run())
            display(run_btn)

    show.on_click(_on_show)
    display(widgets.VBox([show, box]))
