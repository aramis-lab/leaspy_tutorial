"""Generate the self-contained interactive figures (no live kernel needed).

Run this ONCE locally to (re)build the HTML artifacts, then commit them:

    python notebooks/00_animations_interactif.py                  # both figures
    python notebooks/00_animations_interactif.py --sigmoid-only   # skips leaspy
    python notebooks/00_animations_interactif.py path/to/model.json

They are standalone Bokeh documents driven by client-side JS (CustomJS), so
they stay interactive when embedded in a Colab/Jupyter notebook with no Python
kernel recomputing anything.

notebooks/assets/sigmoid_interactive.html
-----------------------------------------
The **average trajectories** of the three Parkinson scores of the canonical
converged fit shipped with the tutorial (``LogisticModel(source_dimension=2)``,
100 000 iterations, on MDS1_total / SCOPA_total / MOCA_total) -- the same curves
as the notebook's

    Plotting(model_100k).average_trajectory(alpha=1, n_std_left=2, n_std_right=8)

same time window, same tab10 colors leaspy's ``Plotting`` gives the features.
No visits are drawn: these are population curves, not patients.

Three sliders -- and *which curves each one moves* is itself the lesson:

    tau  -- onset / time-shift  (individual) : slides ALL THREE curves at once
    g    -- baseline            (population) : p0 = 1/(1+g), MDS1_total only
    v0   -- velocity            (population) : slope at the onset, MDS1_total only

That asymmetry is the model's, not a display choice. ``tau`` is an *individual*
latent variable, drawn once per subject with no feature axis (``tau_mean =
ModelParameter.for_ind_mean("tau", shape=(1,))``, a single scalar): a subject is
early or late *as a whole person*, so there is no per-feature tau to drag. Same
for xi. ``log_g_mean`` and ``log_v0_mean`` in contrast have shape
``(dimension,)``, one entry per feature, so dragging them changes that one score
alone -- exactly as the model can.

Features *can* end up offset from one another in time, but through the sources,
never through tau. In ``model_with_sources`` the space shift enters as

    metric_k * (v0_k * rt + w_k) = metric_k * v0_k * (rt + w_k / v0_k)

so w_k *is* a per-feature time shift of w_k/v0_k years. But ``space_shifts`` are
built on ``OrthoBasis("v0", "metric_sqr")``, hence orthogonal to metric*v0, which
pins those shifts to ``sum_k metric_k * v0_k^2 * delta_k = 0``: advancing one
feature necessarily delays the others. A tau slider moving one curve with the
other two frozen would draw the one thing the geometry forbids.

The individual log-speed **xi is deliberately not a slider**, for two reasons:

* it would be redundant here -- a curve depends on v0 and xi only through the
  product ``v0 * exp(xi)``, so sliding xi by +0.5 is *exactly* multiplying v0
  by e^0.5.  leaspy says as much: ``_center_xi_realizations`` subtracts the mean
  of xi and adds it back to ``log_v0`` at every MCMC-SAEM iteration, "to reduce
  redundancy in the parameter space (i.e. improve identifiability)".  In
  ``SharedSpeedLogisticModel``, where every feature shares one speed, leaspy
  drops ``log_v0_mean`` outright and stores it as ``xi_mean``;
* these are *average* trajectories, and the average trajectory **is** the
  xi = 0 curve.  That is not a convention: ``xi_mean`` is a fixed
  ``Hyperparameter(0.0)``, never estimated, and the centering above forces the
  fitted xi to average to 0 -- whatever speed the cohort has lives in v0.

Pass e.g. ``sliders=("tau", "xi", "g")`` to :func:`build` to get xi back.

The maths mirror leaspy's logistic model. Per feature k, with reparametrized
time rt = exp(xi) * (t - tau) and no sources, ``LogisticModel.model_with_sources``
reduces to the closed-form sigmoid

    p_k(t) = 1 / (1 + g_k * exp( -(g_k+1)^2/g_k * v0_k * exp(xi)*(t - tau) ))

(checked against ``model.compute_mean_traj``: max abs. difference ~1e-7). The
average trajectory is that curve at the mean parameters, tau = tau_mean and
xi = xi_mean = 0.

notebooks/assets/reparam_morph.html
-----------------------------------
The reparametrization animation -- see :func:`build_morph`. Unlike the figure
above it needs leaspy and the dataset (it fits and personalizes a univariate
model to get every patient's real visits); it is skipped if they are missing.
"""

import math
import sys
from pathlib import Path

from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Div, InlineStyleSheet, Slider, Span, Label
from bokeh.plotting import figure
from bokeh.resources import CDN

N = 400  # points along the time axis for the drawn curves

# --- The fit the sigmoid figure draws ----------------------------------------
# The canonical converged run (100 000 iterations, seed 0) shipped in
# content/reference_run/ -- the same model notebooks/fit.ipynb loads. We read its
# saved parameters rather than hardcoding them here, so the figure and the
# notebook can never drift apart. Pass another `model.save(...)` JSON on the CLI
# to rebuild from a different fit.
CANONICAL_MODEL = (
    Path(__file__).resolve().parent.parent
    / "content" / "reference_run" / "parkinson_fit_V21.json"
)
# Index in the model's features of the curve the per-feature sliders (g, v0) drive.
# tau is shared by the whole model, so it always drives every curve.
INTERACTIVE_IX = 0
# Time window, as passed to `average_trajectory` in the notebook: the plot spans
# tau_mean + max(tau_std, 4) * [-N_STD_LEFT, +N_STD_RIGHT].
N_STD_LEFT, N_STD_RIGHT = 2, 8

# matplotlib's tab10, the palette leaspy's `Plotting` defaults to.
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# --- Reparametrization morph (build_morph) ----------------------------------
# Which Parkinson feature to model (univariate). MDS2_total = MDS-UPDRS part II.
FEATURE = "MDS2_total"
AGE_MIN, AGE_MAX = 50.0, 90.0  # window patients must fall in to be picked


def palette(n):
    """The `n` first colors leaspy's `Plotting` hands to the features.

    `Plotting(model).colors()` evaluates `mpl.colormaps["tab10"].resampled(10)`
    at the feature indices, i.e. plain tab10 in feature order. Derived from
    matplotlib when it is installed so the two can't drift apart.
    """
    try:
        import matplotlib as mpl
        from matplotlib.colors import to_hex

        cmap = mpl.colormaps["tab10"].resampled(10)
        return [to_hex(c) for c in cmap(list(range(n)))]
    except Exception:  # matplotlib is optional for this script
        return TAB10[:n]


def fit_facts(model_json=None):
    """Population parameters of the fit -> everything the sigmoid figure draws.

    Defaults to the canonical converged model shipped in ``content/reference_run/``
    (see `CANONICAL_MODEL`); pass the path of another model saved with
    ``model.save(...)`` to use that fit instead. Returns g and v0 per feature on
    their interpretable scale (g = 1/p0 - 1 = exp(log_g), v0 = exp(log_v0)).
    """
    import json

    path = Path(model_json) if model_json is not None else CANONICAL_MODEL
    saved = json.loads(path.read_text(encoding="utf-8"))
    sp = saved["parameters"]
    p = dict(
        features=list(saved["features"]),
        log_g_mean=list(sp["log_g_mean"]),
        log_v0_mean=list(sp["log_v0_mean"]),
        tau_mean=float(sp["tau_mean"][0]),
        tau_std=float(sp["tau_std"][0]),
        xi_std=float(sp["xi_std"][0]),
    )

    return dict(
        features=p["features"],
        colors=palette(len(p["features"])),
        g=[math.exp(x) for x in p["log_g_mean"]],
        v0=[math.exp(x) for x in p["log_v0_mean"]],
        tau_mean=p["tau_mean"],
        tau_std=p["tau_std"],
        xi_std=p["xi_std"],
        source=path.name,
    )


def model_facts():
    """Fit and personalize a univariate logistic on one Parkinson feature and
    return the population params + every patient's fitted (tau, xi) and real
    visits, plus a handful of well-fit patients to spotlight -- the raw material
    of the reparametrization morph. Falls back to synthetic params (which make
    :func:`main` skip the morph) if leaspy / the dataset aren't available."""
    try:
        import warnings

        warnings.filterwarnings("ignore")

        import numpy as np
        from leaspy.datasets import load_dataset
        from leaspy.io.data import Data
        from leaspy.models import LogisticModel

        df = load_dataset("parkinson")[[FEATURE]]
        data = Data.from_dataframe(df)
        model = LogisticModel(name="uni", source_dimension=0, obs_models="gaussian-scalar")
        model.fit(data, "mcmc_saem", seed=0, n_iter=200, progress_bar=False)
        derived = model.compute_derived_parameters()
        g = 1.0 / derived["p0"][0].item() - 1.0
        v0 = derived["v0"][0].item()
        t0 = float(model.parameters["tau_mean"])
        metric = (g + 1) ** 2 / g

        ip = model.personalize(
            data, "scipy_minimize", seed=0, progress_bar=False, use_jacobian=False
        ).to_dataframe()

        def sig(t, tau, xi):
            return 1.0 / (1.0 + g * math.exp(-metric * v0 * math.exp(xi) * (t - tau)))

        # Pool of patients whose visits all fall inside the display window (so
        # nothing flies off-screen), with enough visits to be worth spotlighting.
        allp = []  # (mse, pid, tau, xi)
        for pid, sub in df.groupby(level=0):
            s = sub.reset_index()
            ages = s["TIME"]
            if len(s) < 6 or ages.min() < AGE_MIN + 1 or ages.max() > AGE_MAX - 1:
                continue
            tau, xi = float(ip.loc[pid, "tau"]), float(ip.loc[pid, "xi"])
            mse = float(np.mean([(sig(t, tau, xi) - y) ** 2
                                 for t, y in zip(ages, s[FEATURE])]))
            allp.append((mse, pid, tau, xi))

        # Every patient's fitted (tau, xi) + real visits, so the morph can draw
        # the whole cohort as a faded spaghetti and slide a chosen few onto the
        # reparametrized timeline.
        all_patients = [
            dict(id=pid, tau=float(ip.loc[pid, "tau"]), xi=float(ip.loc[pid, "xi"]),
                 obs_t=sub.reset_index()["TIME"].tolist(),
                 obs_y=sub.reset_index()[FEATURE].tolist())
            for pid, sub in df.groupby(level=0)
        ]
        # Highlight a handful of WELL-FIT patients spread across onset age (tau),
        # so the animation shows early/late & slow/fast subjects all converging
        # onto the same population curve. `allp` is already the in-window pool.
        good = sorted(allp)[: max(8, len(allp) // 2)]   # lowest-mse half
        good_by_tau = sorted(good, key=lambda q: q[2])   # ordered by tau
        k = min(5, len(good_by_tau))
        picks = ([good_by_tau[round(i * (len(good_by_tau) - 1) / (k - 1))] for i in range(k)]
                 if k > 1 else good_by_tau)
        highlight_ids = [q[1] for q in picks]
        # Always showcase the patient reaching the HIGHEST observed score -- the
        # dramatic steep riser (also the fastest). After reparametrization it
        # stretches far along disease time, landing high on the population curve.
        top_score = max(allp, key=lambda q: float(df.loc[q[1], FEATURE].max()))[1]
        if top_score not in highlight_ids:
            highlight_ids.append(top_score)

        return dict(source="real", feature=FEATURE, g=g, v0=v0, t0=t0,
                    all_patients=all_patients, highlight_ids=highlight_ids)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"(real-data fit unavailable: {exc!r} -- morph will be skipped)")
        return dict(source="synthetic", feature="score", g=5.0, v0=0.01, t0=68.0)


def sigmoid(t, tau, xi, *, g, v0):
    """leaspy's logistic trajectory of one feature, sampled at the ages `t`."""
    metric = (g + 1) ** 2 / g
    a = math.exp(xi)
    return [1.0 / (1.0 + g * math.exp(-metric * v0 * a * (ti - tau))) for ti in t]


def build(f, *, sliders=("tau", "g", "v0")):
    """Build the standalone HTML: the average trajectory of every feature of the
    fit. tau/xi reparametrize time for the whole model, so they move every curve;
    g and v0 are per-feature, so they move the ``INTERACTIVE_IX``-th one alone.

    ``sliders`` picks which of "tau", "xi", "g", "v0" are exposed, in order.
    The default leaves "xi" out on purpose -- see the module docstring: it is
    redundant with v0 on a single curve, and the average trajectory *is* the
    xi = 0 curve. The four Slider objects are always built (the callback reads
    all of them); only the requested ones make it into the layout.
    """
    feats, colors, gs, v0s = f["features"], f["colors"], f["g"], f["v0"]
    t0, tau_std = f["tau_mean"], f["tau_std"]
    ix = INTERACTIVE_IX

    # Same age window as `Plotting.average_trajectory` draws it in fit.ipynb:
    # tau_mean + max(tau_std, 4) * [-n_std_left, n_std_right].
    span = max(tau_std, 4.0)
    t_min, t_max = t0 - N_STD_LEFT * span, t0 + N_STD_RIGHT * span

    # Geometry: we fix the *data frame* size (not total width). The legend is
    # added outside on the right, so it extends the total width without shrinking
    # the frame -- the sliders (width = frame width, left margin = BORDER_L) stay
    # aligned with the plot. Bump BORDER_L if the τ handle sits a hair left.
    FRAME_W, FRAME_H = 600, 350
    BORDER_L, BORDER_R = 62, 8

    t = [t_min + (t_max - t_min) * i / (N - 1) for i in range(N)]

    # Slider grids. The fitted values are the sliders' starting point, their
    # magnet and what "reset" returns to, so snap them onto the grid once here:
    # a magnet off the grid could not be reached by dragging.
    lo = dict(tau=t_min, xi=-1.5, g=0.2, v0=0.001)
    # ξ spans ~ ±2-3 fitted xi_std; g and v0 stretch well past the fit (and are
    # widened if another fit lands outside), enough to deform the curve a lot.
    hi = dict(tau=t_max, xi=2.5, g=max(12.0, 1.5 * gs[ix]),
              v0=max(0.03, 2.5 * v0s[ix]))
    step = dict(tau=0.1, xi=0.02, g=0.05, v0=0.0002)

    def snap(key, value):
        return lo[key] + round((value - lo[key]) / step[key]) * step[key]

    fitted = dict(tau=snap("tau", t0), xi=snap("xi", 0.0),
                  g=snap("g", gs[ix]), v0=snap("v0", v0s[ix]))

    # Average trajectory of each feature = the model at the mean individual
    # parameters (tau = tau_mean, xi = xi_mean = 0, sources = 0). EVERY feature
    # gets a live source: tau/xi are one scalar for the whole subject, so a drag
    # on tau has to slide the three curves together. g and v0 carry a feature
    # axis, so only feature `ix` reads them from the sliders -- the others keep
    # their fitted (gs[k], v0s[k]).
    live_srcs = [
        ColumnDataSource(dict(
            t=t, y=sigmoid(t, fitted["tau"], fitted["xi"],
                           g=fitted["g"] if k == ix else gs[k],
                           v0=fitted["v0"] if k == ix else v0s[k])))
        for k in range(len(feats))
    ]

    p = figure(
        x_axis_label="Age",
        y_axis_label="Normalized score",
        frame_width=FRAME_W, frame_height=FRAME_H,
        # leaspy clamps logistic plots to (0, 1); a hair of padding keeps the
        # saturated ends of the curves from being cut in half by the frame.
        x_range=(t_min, t_max), y_range=(-0.015, 1.015),
        min_border_left=BORDER_L, min_border_right=BORDER_R,
        tools="", toolbar_location=None,
    )

    # The features, in model order and in leaspy's own colors.
    for k, (name, col) in enumerate(zip(feats, colors)):
        p.line("t", "y", source=live_srcs[k], color=col, line_width=4,
               legend_label=name)

    # Onset marker: tau belongs to the subject, not to any one score, so the line
    # is neutral grey rather than the interactive feature's color.
    onset = Span(location=fitted["tau"], dimension="height",
                 line_color="#555555", line_dash="dotted", line_width=1.5)
    p.add_layout(onset)

    # α = exp(ξ) is only worth reading out when ξ is one of the sliders --
    # otherwise it is pinned to 1 and says nothing.
    show_alpha = "xi" in sliders

    def readout_text(xi, g):
        alpha = f"α = exp(ξ) = {math.exp(xi):.2f}      " if show_alpha else ""
        return f"{alpha}p₀ = 1/(1+g) = {1 / (1 + g):.3f}"

    # Opaque background: the onset line is full-height and would otherwise be
    # drawn straight through the text whenever τ sits under it.
    readout = Label(x=t_min + 0.02 * (t_max - t_min), y=0.93,
                    text_font_size="11pt", text_color=colors[ix],
                    text=readout_text(fitted["xi"], fitted["g"]),
                    background_fill_color="white", background_fill_alpha=0.85,
                    padding=3)
    p.add_layout(readout)

    # Shown only when g = 1: a faint midline at 0.5 on the plot + a note (a Div in
    # the layout, reserved height so it doesn't shift the sliders) explaining that
    # the onset τ then sits at the trajectory's midpoint.
    mid_line = Span(location=0.5, dimension="width", line_color="#9e9e9e",
                    line_dash="dotted", line_width=1.2, visible=False)
    p.add_layout(mid_line)
    mid_div = Div(text="", height=22,
                  styles={"font-size": "11px", "color": "#444",
                          "padding-left": f"{BORDER_L}px"})

    # Move the legend outside to the right and make it a clickable table.
    legend = p.legend[0]
    legend.title = "Features"           # same legend title as leaspy's Plotting
    legend.click_policy = "hide"
    legend.label_text_font_size = "9pt"
    p.add_layout(legend, "right")

    # Sliders. Width = frame width, left margin = left border -> aligned track.
    smargin = (4, 0, 4, BORDER_L)

    def make(key, title, **kw):
        # `name` is only there to make the sliders reachable from the console /
        # a test script: doc.get_model_by_name("tau").value = 90
        return Slider(name=key, start=lo[key], end=hi[key], value=fitted[key],
                      step=step[key], title=title, width=FRAME_W,
                      margin=smargin, **kw)

    # Titles say which *level* each parameter lives at, because that is exactly
    # what decides whether the slider moves one curve or all three.
    tau_slider = make("tau", "onset  τ   (individual — time-shift, all features)")
    xi_slider = make("xi", "speed  ξ   (individual — log-acceleration, α = exp ξ)")
    g_slider = make("g", f"baseline  g   (population — p₀ = 1/(1+g), {feats[ix]})")
    v0_slider = make("v0", f"velocity  v₀   (population — slope at the onset, "
                           f"{feats[ix]})", format="0.0000")

    def magnet(slider, points, tol):
        """Snap the slider to any of `points` when dragged within `tol` of it."""
        slider.js_on_change("value", CustomJS(
            args=dict(s=slider, M=list(points), tol=tol),
            code="for (let k=0;k<M.length;k++){"
                 "if (Math.abs(s.value-M[k])<=tol && s.value!==M[k]){s.value=M[k];break;}}"))

    # Magnets: register BEFORE the recompute callback so the snapped value is the
    # one used to redraw the curve. g has two: the special g=1 and the fitted g.
    g_magnets = [1.0, fitted["g"]]
    magnet(tau_slider, [fitted["tau"]], 0.5)
    magnet(xi_slider, [fitted["xi"]], 0.1)
    magnet(g_slider, g_magnets, 0.2)
    magnet(v0_slider, [fitted["v0"]], 0.0004)

    # Gray dots on the track marking each magnet (default values). Bokeh 3 renders
    # the slider inside a shadow root, so we inject CSS *into* it via stylesheets;
    # the noUi track elements are reachable there. Up to two dots (::before/::after).
    def magnet_dots(slider, magnets, key):
        dot = ("content:'';position:absolute;top:50%;width:9px;height:9px;"
               "margin:-4.5px 0 0 -4.5px;border-radius:50%;background:#9e9e9e;"
               "pointer-events:none;z-index:5;")
        css = ".noUi-target{position:relative;}"
        for m, ps in zip(magnets, ("after", "before")):
            css += ".noUi-target::%s{%sleft:%.3f%%;}" % (
                ps, dot, (m - lo[key]) / (hi[key] - lo[key]) * 100)
        slider.stylesheets = [InlineStyleSheet(css=css)]

    magnet_dots(tau_slider, [fitted["tau"]], "tau")
    magnet_dots(xi_slider, [fitted["xi"]], "xi")
    magnet_dots(g_slider, g_magnets, "g")
    magnet_dots(v0_slider, [fitted["v0"]], "v0")

    # One callback redraws every feature. τ/ξ reparametrize the time axis shared
    # by all of them; g/v0 reshape feature `ix` only, the others keeping the
    # fitted GS[k]/V0S[k] -- which is precisely the model's own asymmetry.
    callback = CustomJS(
        args=dict(srcs=live_srcs, ix=ix, GS=list(gs), V0S=list(v0s),
                  onset=onset, readout=readout, mid_line=mid_line,
                  mid_div=mid_div, tau=tau_slider, xi=xi_slider, g=g_slider,
                  v0=v0_slider, show_alpha=show_alpha),
        code="""
        const G = g.value, A = Math.exp(xi.value), TAU = tau.value;
        for (let k = 0; k < srcs.length; k++) {
            const Gk = (k === ix) ? G : GS[k];
            const Vk = (k === ix) ? v0.value : V0S[k];
            const metric = Math.pow(Gk + 1, 2) / Gk;
            const t = srcs[k].data['t'], y = srcs[k].data['y'];
            for (let i = 0; i < t.length; i++)
                y[i] = 1 / (1 + Gk * Math.exp(-metric * Vk * A * (t[i] - TAU)));
            srcs[k].change.emit();
        }
        onset.location = TAU;
        readout.text = (show_alpha ? "α = exp(ξ) = " + A.toFixed(2) + "      " : "")
                     + "p₀ = 1/(1+g) = " + (1 / (1 + G)).toFixed(3);
        const showMid = Math.abs(G - 1) < 0.06;
        mid_line.visible = showMid;
        mid_div.text = showMid
            ? "<b>g = 1</b> → p(τ) = 0.5 : at the onset τ the trajectory is exactly at its midpoint."
            : "";
        """,
    )
    by_key = dict(tau=tau_slider, xi=xi_slider, g=g_slider, v0=v0_slider)
    shown = [by_key[k] for k in sliders]
    for s in shown:
        s.js_on_change("value", callback)

    return file_html(column(p, mid_div, *shown), CDN,
                     "leaspy interactive sigmoid")


# Self-running autoplay script injected into the standalone HTML. It drives the
# Bokeh models directly (no kernel, no widgets): hold on the raw plot, animate
# the whole cohort onto the reparametrized timeline, hold on the result, repeat.
def _rainbow_hex(u, s=0.62, v=0.85):
    """Map u in [0, 1] to a pleasant rainbow hex color (HSV, hue 0->0.83)."""
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(0.83 * max(0.0, min(1.0, u)), s, v)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


_MORPH_TEMPLATE = r"""
{% block postamble %}
<script>
(function () {
  // Phase durations (ms). The whole thing autoplays and loops forever.
  const T_HOLD = 1000,    // 1. everyone in color, raw ages
        T_HILITE = 3000,  // 2. crowd fades to grey, chosen patients spotlighted
        T_MORPH = 3500,    // 3. the whole cohort slides onto the timeline
        T_REST = 1400,     // 4. hold on the reparametrized result
        T_RECOLOR = 1300,  // 5. the grey crowd comes back to color
        T_FINAL = 600;     // 6. hold on the colored, aligned cohort
  const TOTAL = T_HOLD + T_HILITE + T_MORPH + T_REST + T_RECOLOR + T_FINAL;
  const GREY = "#9e9e9e";

  const smooth = (u) => u * u * (3 - 2 * u);
  const clamp = (u) => Math.max(0, Math.min(1, u));
  function lerpHex(a, b, t) {                 // a,b "#rrggbb" -> blended hex
    const A = parseInt(a.slice(1), 16), B = parseInt(b.slice(1), 16);
    const r = Math.round((A >> 16) + (((B >> 16) - (A >> 16)) * t));
    const g = Math.round(((A >> 8) & 255) + ((((B >> 8) & 255) - ((A >> 8) & 255)) * t));
    const b2 = Math.round((A & 255) + (((B & 255) - (A & 255)) * t));
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b2).toString(16).slice(1);
  }

  function start() {
    if (!window.Bokeh || !Bokeh.documents || !Bokeh.documents.length)
      return setTimeout(start, 80);
    const doc = Bokeh.documents[0], M = (n) => doc.get_model_by_name(n);
    const bg = M('bg_src'), hl = M('hl_src'), dots = M('dot_src'),
          bgr = M('bg_r'), pop = M('pop_r'), stage = M('stage');
    if (!bg || !hl || !dots || !bgr || !pop || !stage)
      return setTimeout(start, 80);
    const colorOn = bg.data['color_on'];

    function morphMulti(src, e) {              // interpolate a multi_line's x-coords
      const xs = src.data['xs'], xr = src.data['xs_raw'], xp = src.data['xs_rep'];
      for (let k = 0; k < xs.length; k++) {
        const a = xr[k], b = xp[k], o = xs[k];
        for (let j = 0; j < o.length; j++) o[j] = (1 - e) * a[j] + e * b[j];
      }
    }

    // Render one frame from the global state (e=morph, c=colorfulness 1..0, a=alpha).
    function render(e, c, alpha, popA, label) {
      morphMulti(hl, e);
      morphMulti(bg, e);
      const col = bg.data['color'];
      for (let k = 0; k < col.length; k++) col[k] = lerpHex(GREY, colorOn[k], c);
      bg.change.emit();
      hl.change.emit();
      const gx = dots.data['x'], gr = dots.data['x_raw'], gp = dots.data['x_rep'];
      for (let i = 0; i < gx.length; i++) gx[i] = (1 - e) * gr[i] + e * gp[i];
      dots.change.emit();
      bgr.glyph.line_alpha = alpha;
      pop.glyph.line_alpha = 0.85 * popA;
      stage.text = label;
    }

    const L_RAW = "raw ages — the whole cohort, in color";
    const L_HI = "spotlighting a few patients (the rest fade to grey)";
    const L_MO = "reparametrizing:  ψ(t) = exp(ξ)·(t − τ) + τ̄";
    const L_REP = "reparametrized — every patient aligns on the population curve";

    // Pause/Resume button, placed in normal flow at the very top so it shows
    // regardless of how the standalone doc is embedded. The loop accumulates
    // `elapsed` only while running, so pausing freezes and resuming continues
    // without a jump.
    let elapsed = 0, last = performance.now(), paused = false;
    const bar = document.createElement('div');
    bar.style.cssText = 'text-align:center;margin:8px 0 4px;';
    const btn = document.createElement('button');
    btn.textContent = '⏸ Pause';
    btn.style.cssText = 'padding:7px 20px;font:14px sans-serif;color:#fff;' +
      'background:#1f77b4;border:none;border-radius:6px;cursor:pointer;' +
      'box-shadow:0 1px 3px rgba(0,0,0,.2);';
    btn.onclick = function () {
      paused = !paused;
      btn.textContent = paused ? '▶ Resume' : '⏸ Pause';
      btn.style.background = paused ? '#2ca02c' : '#1f77b4';
    };
    bar.appendChild(btn);
    document.body.insertBefore(bar, document.body.firstChild);

    function frame(now) {
      if (!paused) elapsed += now - last;
      last = now;
      const t = elapsed % TOTAL;
      let a = T_HOLD, b = a + T_HILITE, cM = b + T_MORPH, d = cM + T_REST, eP = d + T_RECOLOR;
      if (t < a) {                                   // 1. colored, raw
        render(0, 1, 0.60, 0, L_RAW);
      } else if (t < b) {                            // 2. fade crowd to grey
        const u = (t - a) / T_HILITE;
        render(0, 1 - u, 0.60 + (0.16 - 0.60) * u, 0, L_HI);
      } else if (t < cM) {                           // 3. morph onto the curve
        const u = (t - b) / T_MORPH;
        render(smooth(u), 0, 0.16, clamp(u / 0.4), L_MO);
      } else if (t < d) {                            // 4. hold reparametrized
        render(1, 0, 0.16, 1, L_REP);
      } else if (t < eP) {                           // 5. crowd back to color
        const u = (t - d) / T_RECOLOR;
        render(1, u, 0.16 + (0.60 - 0.16) * u, 1, L_REP);
      } else {                                       // 6. hold colored & aligned
        render(1, 1, 0.60, 1, L_REP);
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  start();
})();
</script>
{% endblock %}
"""


def build_morph(f):
    """Standalone, self-playing HTML telling the reparametrization story:

    1. the whole cohort as a raw spaghetti plot (visits at real ages),
    2. a few highlighted (colored) patients while the grey crowd fades,
    3. EVERY patient smoothly slides from real age to *reparametrized* age
       ψᵢ(t) = exp(ξᵢ)·(t − τᵢ) + tau_mean, the cohort collapsing onto the shared
       population sigmoid -- the geometric meaning of personalization.

    The animation autoplays and loops (3 s hold → morph → 3 s hold → replay),
    driven by an injected script that talks to the Bokeh models directly, so no
    kernel and no widgets are needed. Only x-positions move; y-values are data.
    """
    import math

    g, v0, t0 = f["g"], f["v0"], f["t0"]
    metric = (g + 1) ** 2 / g
    allp = f["all_patients"]
    hl_ids = list(f["highlight_ids"])
    hl_set = set(hl_ids)
    by_id = {p["id"]: p for p in allp}

    def reparam(p):
        a = math.exp(p["xi"])
        return [a * (t - p["tau"]) + t0 for t in p["obs_t"]]

    # Background crowd: keep patients whose visits AND reparametrized ages both
    # stay in a sensible window (so nothing flies off-screen when they morph),
    # then subsample to ~70 lines -- a readable cohort without bloating the HTML.
    AGE_W = (45.0, 92.0)
    REP_W = (t0 - 28.0, t0 + 30.0)
    bg = []
    for p in allp:
        if p["id"] in hl_set or not (AGE_W[0] <= min(p["obs_t"]) and max(p["obs_t"]) <= AGE_W[1]):
            continue
        rp = reparam(p)
        if REP_W[0] <= min(rp) and max(rp) <= REP_W[1]:
            bg.append((p, rp))
    bg = bg[:: max(1, len(bg) // 70)]
    bg_xs = [p["obs_t"] for p, _ in bg]
    bg_rep = [rp for _, rp in bg]
    bg_ys = [p["obs_y"] for p, _ in bg]
    # A colorful crowd: hue spread across onset age (tau) -> a rainbow spaghetti
    # that the animation desaturates to grey to spotlight the chosen patients.
    bg_taus = [p["tau"] for p, _ in bg]
    tlo, thi = (min(bg_taus), max(bg_taus)) if bg_taus else (0.0, 1.0)
    bg_color_on = [_rainbow_hex((tau - tlo) / (thi - tlo) if thi > tlo else 0.5)
                   for tau in bg_taus]

    # Highlighted patients (drawn on top, colored). One color each.
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    hl = [by_id[i] for i in hl_ids]
    hl_colors = [palette[i % len(palette)] for i in range(len(hl))]
    xs_raw = [list(p["obs_t"]) for p in hl]
    xs_rep = [reparam(p) for p in hl]
    ys = [list(p["obs_y"]) for p in hl]

    # x-range must cover both raw ages and reparametrized ages of everything shown.
    allx = [x for ser in (bg_xs + bg_rep + xs_raw + xs_rep) for x in ser]
    xpad = 0.04 * (max(allx) - min(allx))
    x_min, x_max = min(allx) - xpad, max(allx) + xpad
    ally = [y for ser in (bg_ys + ys) for y in ser]
    y_min, y_max = min(ally) - 0.04, max(ally) + 0.06

    # Population-average sigmoid over the reparametrized-age axis (xi=0, tau=t0).
    tt = [x_min + (x_max - x_min) * i / (N - 1) for i in range(N)]
    pop_y = [1.0 / (1.0 + g * math.exp(-metric * v0 * (ti - t0))) for ti in tt]
    pop_src = ColumnDataSource(dict(t=tt, y=pop_y))

    # Sources carry both endpoints (xs_raw, xs_rep) + a live `xs` the script
    # interpolates. Names let the injected script find them via the document.
    bg_src = ColumnDataSource(name="bg_src", data=dict(
        xs=[list(x) for x in bg_xs], ys=bg_ys, xs_raw=bg_xs, xs_rep=bg_rep,
        color=list(bg_color_on), color_on=bg_color_on))
    hl_line_src = ColumnDataSource(name="hl_src", data=dict(
        xs=[list(x) for x in xs_raw], ys=ys, xs_raw=xs_raw, xs_rep=xs_rep, color=hl_colors))
    dx, dxr, dxp, dy, dc = [], [], [], [], []
    for xr, xp, yv, c in zip(xs_raw, xs_rep, ys, hl_colors):
        for a, b, yy in zip(xr, xp, yv):
            dx.append(a); dxr.append(a); dxp.append(b); dy.append(yy); dc.append(c)
    hl_dot_src = ColumnDataSource(name="dot_src",
                                  data=dict(x=dx, x_raw=dxr, x_rep=dxp, y=dy, color=dc))

    p = figure(
        title=f"Reparametrization — Parkinson {f['feature']}: raw ages → shared disease timeline",
        x_axis_label="age (years)   →   reparametrized disease time",
        y_axis_label=f"normalized {f['feature']}",
        frame_width=720, frame_height=420,
        x_range=(x_min, x_max), y_range=(y_min, y_max),
        min_border_left=62, min_border_right=12,
        tools="", toolbar_location=None,
    )

    pop_r = p.line("t", "y", source=pop_src, line_width=3, color="#444444",
                   line_dash="dashed", line_alpha=0.0)
    pop_r.name = "pop_r"
    # Keep the legend sample independent from the animated population curve.
    # An empty renderer draws nothing in the plot while providing a stable key.
    legend_src = ColumnDataSource(dict(t=[], y=[]))
    p.line("t", "y", source=legend_src, line_width=3, color="#444444",
           line_dash="dashed", line_alpha=0.85,
           legend_label="population average")
    bg_r = p.multi_line("xs", "ys", source=bg_src, line_color="color",
                        line_alpha=0.60, line_width=1.3)
    bg_r.name = "bg_r"
    p.multi_line("xs", "ys", source=hl_line_src, line_color="color",
                 line_width=2.5, line_alpha=0.95)
    p.scatter("x", "y", source=hl_dot_src, color="color", size=8,
              line_color="white", line_width=0.6)

    stage = Label(name="stage", x=x_min + 0.02 * (x_max - x_min), y=y_max - 0.05,
                  text="raw ages — highlighting a few patients (others fade)",
                  text_font_size="11pt", text_color="#555555")
    p.add_layout(stage)
    p.legend.location = "bottom_right"
    p.legend.label_text_font_size = "9pt"

    return file_html(column(p), CDN, "leaspy reparametrization morph",
                     template=_MORPH_TEMPLATE)


def main(model_json=None, *, sigmoid=True, morph=True):
    assets = Path(__file__).resolve().parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # Average trajectories of the canonical fit -- needs no leaspy, just the model JSON
    # (or the population parameters of the model JSON passed on the CLI).
    if sigmoid:
        facts = fit_facts(model_json)
        out = assets / "sigmoid_interactive.html"
        out.write_text(build(facts), encoding="utf-8")
        fts = ", ".join(f"{n} (g={g:.2f}, v0={v:.4f})"
                        for n, g, v in zip(facts["features"], facts["g"], facts["v0"]))
        print(f"wrote {out}  (params from {facts['source']}: "
              f"tau_mean={facts['tau_mean']:.2f} | {fts})")

    # Reparametrization morph (raw spaghetti → reparametrized, animated). This
    # one refits from the dataset, so it is skipped without leaspy.
    if morph:
        cohort = model_facts()
        if cohort["source"] == "real" and cohort.get("all_patients"):
            out2 = assets / "reparam_morph.html"
            out2.write_text(build_morph(cohort), encoding="utf-8")
            print(f"wrote {out2}  ({cohort['feature']}, "
                  f"highlighted: {', '.join(cohort['highlight_ids'])})")


if __name__ == "__main__":
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    path = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    main(path,
         sigmoid="--morph-only" not in flags,
         morph="--sigmoid-only" not in flags)
