"""Generate a self-contained interactive sigmoid figure (no live kernel needed).

Run this ONCE locally to (re)build the HTML artifact, then commit it:

    python notebooks/00_animations_interactif.py

It writes  notebooks/assets/sigmoid_interactive.html  -- a standalone Bokeh
document whose sliders are driven by client-side JS (CustomJS), so the figure
stays interactive when embedded in a Colab/Jupyter notebook with no Python
kernel recomputing anything.

What it shows
-------------
A *real* leaspy workflow on one feature of the synthetic Parkinson dataset
(``leaspy.datasets.load_dataset("parkinson")``, same data as the gallery's
``plot_02_parkinson_example``):

* grey dashed curve  -- the fitted **population-average** trajectory (the curve
  at the mean parameters: tau = tau_mean, xi = 0),
* three real patients, each shown as their fitted logistic curve + their real
  visit dots:
    - the **main** patient -- driven by the sliders (this is personalization),
    - an **earlier & slower** patient (opposite corner from the main one),
    - an **≈ average** patient (tau ~ tau_mean, xi ~ 0) whose curve nearly
      overlaps the grey line -- making it intuitive that the population curve is
      the trajectory of the *average-parameter* patient.

Click a legend entry to hide/show that patient (Bokeh interactive legend).

The two main sliders are the main patient's individual parameters; g is the
shared population baseline. Drag tau/xi to *see what personalization does*: find
the (tau, xi) that make the curve pass through the patient's dots.

    tau  -- onset / time-shift   : slides the curve left <-> right
    xi   -- log-speed            : alpha = exp(xi) compresses/stretches time
    g    -- population baseline  : p0 = 1/(1+g); reshapes every curve

The maths mirror leaspy's univariate logistic model. For one patient the
trajectory is the closed-form sigmoid (LogisticModel + reparametrized time
rt = exp(xi) * (t - tau)):

    p(t) = 1 / (1 + g * exp( -(g+1)^2/g * v0 * exp(xi)*(t - tau) ))

where the population parameters g (= 1/p0 - 1) and v0, and the reference age
tau_mean, all come from the fitted model.

If leaspy or the dataset are unavailable, the script falls back to a single
synthetic patient so it still produces a usable figure.
"""

from pathlib import Path

from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, CustomJS, Div, InlineStyleSheet, Slider, Span, Label
from bokeh.plotting import figure
from bokeh.resources import CDN

# Which Parkinson feature to model (univariate). MDS2_total = MDS-UPDRS part II.
FEATURE = "MDS2_total"
N = 400  # points along the time axis for the drawn curves
AGE_MIN, AGE_MAX = 50.0, 90.0  # fixed display window (also gates patient picks)

# Per-role styling.
COLOR = {"main": "#1f77b4", "opp": "#ff7f0e", "avg": "#2ca02c"}


def _label(pt):
    return {
        "main": f"{pt['id']} — this patient (sliders)",
        "opp": f"{pt['id']} — earlier & slower",
        "avg": f"{pt['id']} — ≈ average",
    }[pt["role"]]


def model_facts():
    """Fit a univariate logistic on one Parkinson feature and return the real
    population params + three illustrative patients (main / opposite / average),
    each with their fitted (tau, xi) and real visits. Falls back to a single
    synthetic patient if leaspy / the dataset aren't available."""
    try:
        import warnings

        warnings.filterwarnings("ignore")
        import math

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

        # Two pools, both restricted to patients whose visits fall inside the
        # display window (so their dots are visible). `allp` (>=6 visits) is broad
        # enough to contain patients with xi ~ 0; `mainp` additionally needs a
        # clear progression so the slider-controlled patient visibly rises.
        allp, mainp = [], []  # (mse, pid, tau, xi)
        for pid, sub in df.groupby(level=0):
            s = sub.reset_index()
            ages = s["TIME"]
            if len(s) < 6 or ages.min() < AGE_MIN + 1 or ages.max() > AGE_MAX - 1:
                continue
            tau, xi = float(ip.loc[pid, "tau"]), float(ip.loc[pid, "xi"])
            mse = float(np.mean([(sig(t, tau, xi) - y) ** 2
                                 for t, y in zip(ages, s[FEATURE])]))
            allp.append((mse, pid, tau, xi))
            if len(s) >= 8 and (s[FEATURE].max() - s[FEATURE].min()) >= 0.3:
                mainp.append((mse, pid, tau, xi))
        main_mse, main_id, main_tau, main_xi = min(mainp)

        # Among the decently-fit half, pick the patient nearest each target point
        # in (tau, xi) space (5 yr ~ 1 xi unit). "avg" sits on the population
        # curve (tau_mean, 0); "opp" mirrors the main patient across that mean.
        med = float(np.median([q[0] for q in allp]))
        decent = [q for q in allp if q[0] <= med]

        def nearest(tt, tx, exclude):
            cand = [q for q in decent if q[1] not in exclude]
            return min(cand, key=lambda q: ((q[2] - tt) / 5.0) ** 2 + (q[3] - tx) ** 2)[1]

        # "avg" sits on the population curve (tau_mean, xi~0). "opp" is clearly
        # earlier AND slower than average (the opposite corner from a late/fast
        # main patient) and well-fit -- picked as the best-fitting such patient so
        # it stays visually distinct from both the average and the main patient.
        avg_id = nearest(t0, 0.0, {main_id})
        opp = [q for q in allp if q[2] <= t0 - 3 and q[3] <= -0.25
               and q[1] not in {main_id, avg_id}]
        opp_id = (min(opp)[1] if opp
                  else nearest(2 * t0 - main_tau, -main_xi, {main_id, avg_id}))

        def record(pid, role):
            s = df.loc[pid].reset_index()
            return dict(id=pid, role=role,
                        tau=float(ip.loc[pid, "tau"]), xi=float(ip.loc[pid, "xi"]),
                        obs_t=s["TIME"].tolist(), obs_y=s[FEATURE].tolist())

        # Order matters: index 0 is the slider-controlled patient.
        patients = [record(main_id, "main"), record(opp_id, "opp"), record(avg_id, "avg")]

        # --- Extra data for the reparametrization morph (build_morph) ---------
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
                    patients=patients, all_patients=all_patients,
                    highlight_ids=highlight_ids)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"(real-data fit unavailable: {exc!r} -- using synthetic params)")
        return dict(
            source="synthetic", feature="score", g=5.0, v0=0.01, t0=68.0,
            patients=[dict(id="synthetic", role="main", tau=66.0, xi=0.3,
                           obs_t=[], obs_y=[])],
        )


def sigmoid(t, tau, xi, *, g, v0):
    import math

    metric = (g + 1) ** 2 / g
    a = math.exp(xi)
    return [1.0 / (1.0 + g * math.exp(-metric * v0 * a * (ti - tau))) for ti in t]


def build(f, *, show_g=False, show_v0=False):
    """Build the standalone HTML. ``show_g`` / ``show_v0`` expose the population
    shape parameters as extra sliders (they reshape every curve)."""
    g, v0, t0 = f["g"], f["v0"], f["t0"]
    patients = f["patients"]
    main = patients[0]

    # Fixed age axis.
    t_min, t_max = AGE_MIN, AGE_MAX

    # Geometry: we fix the *data frame* size (not total width). The legend is
    # added outside on the right, so it extends the total width without shrinking
    # the frame -- the sliders (width = frame width, left margin = BORDER_L) stay
    # aligned with the plot. Bump BORDER_L if the τ handle sits a hair left.
    FRAME_W, FRAME_H = 600, 350
    BORDER_L, BORDER_R = 62, 8

    t = [t_min + (t_max - t_min) * i / (N - 1) for i in range(N)]
    # One curve source per patient (aligned with `patients`; index 0 = main).
    curve_srcs = [ColumnDataSource(dict(t=t, y=sigmoid(t, pt["tau"], pt["xi"], g=g, v0=v0)))
                  for pt in patients]
    fixed_tau = [pt["tau"] for pt in patients]
    fixed_xi = [pt["xi"] for pt in patients]
    mean_src = ColumnDataSource(dict(t=t, y=sigmoid(t, t0, 0.0, g=g, v0=v0)))

    if f["source"] == "real":
        title = f"leaspy logistic on real data — Parkinson {f['feature']} (3 patients)"
    else:
        title = "leaspy univariate logistic — move one patient with τ and speed"

    p = figure(
        title=title,
        x_axis_label="age (years)",
        y_axis_label=f"normalized {f['feature']}",
        frame_width=FRAME_W, frame_height=FRAME_H,
        x_range=(t_min, t_max), y_range=(-0.02, 1.02),
        min_border_left=BORDER_L, min_border_right=BORDER_R,
        tools="", toolbar_location=None,
    )

    # Population average (drawn first, sits underneath).
    p.line("t", "y", source=mean_src, line_width=2, line_dash="dashed",
           color="#9e9e9e", legend_label="population average")

    # Patients: extras first (50% alpha), main last so it sits on top. Each
    # patient's curve and dots share a legend label -> one click toggles both.
    draw_order = [pt for pt in patients if pt["role"] != "main"] + [main]
    for pt in draw_order:
        i = patients.index(pt)
        is_main = pt["role"] == "main"
        col, lab = COLOR[pt["role"]], _label(pt)
        # Extras are 80% transparent (alpha 0.2) so the blue main patient stands out.
        a = 1.0 if is_main else 0.2
        p.line("t", "y", source=curve_srcs[i], color=col, legend_label=lab,
               line_width=3 if is_main else 2.5, line_alpha=a)
        if pt["obs_t"]:
            p.scatter("t", "y",
                      source=ColumnDataSource(dict(t=pt["obs_t"], y=pt["obs_y"])),
                      color=col, legend_label=lab,
                      size=8 if is_main else 7,
                      fill_alpha=0.55 if is_main else 0.2, line_alpha=a)

    # Onset marker + alpha readout track the MAIN patient.
    onset = Span(location=main["tau"], dimension="height", line_color=COLOR["main"],
                 line_dash="dotted", line_width=1.5)
    p.add_layout(onset)
    alpha_label = Label(x=t_min + 1, y=0.92, text_font_size="11pt",
                        text_color=COLOR["main"],
                        text=f"α = exp(ξ) = {2.718281828 ** main['xi']:.2f}")
    p.add_layout(alpha_label)

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
    legend.click_policy = "hide"
    legend.label_text_font_size = "9pt"
    p.add_layout(legend, "right")

    # Sliders. Width = frame width, left margin = left border -> aligned track.
    # Steps are chosen so the magnet values land exactly on the slider grid.
    smargin = (4, 0, 4, BORDER_L)
    mtau, mxi = round(main["tau"], 1), round(main["xi"], 2)  # GS-001's fitted values

    def magnet(slider, points, tol):
        """Snap the slider to any of `points` when dragged within `tol` of it."""
        slider.js_on_change("value", CustomJS(
            args=dict(s=slider, M=list(points), tol=tol),
            code="for (let k=0;k<M.length;k++){"
                 "if (Math.abs(s.value-M[k])<=tol && s.value!==M[k]){s.value=M[k];break;}}"))

    tau_slider = Slider(start=t_min, end=t_max, value=main["tau"], step=0.1,
                        title="onset  τ  (time-shift)", width=FRAME_W, margin=smargin)
    xi_slider = Slider(start=-1.5, end=2.5, value=main["xi"], step=0.02,
                       title="speed  ξ  (log-acceleration)", width=FRAME_W, margin=smargin)
    # g and v0 are population (shape) params: exposed as extra sliders via
    # show_g / show_v0. Either way the JS reads `.value`, so we always hand the
    # callback real Slider objects but only add the requested ones to the layout.
    g_slider = Slider(start=0.2, end=12.0, value=g, step=0.05,
                      title="baseline  g   (p₀ = 1/(1+g))",
                      width=FRAME_W, margin=smargin)
    v0_slider = Slider(start=0.001, end=0.05, value=v0, step=0.001, format="0.000",
                       title="velocity  v₀  (slope at onset)", width=FRAME_W, margin=smargin)

    # Magnets: register BEFORE the recompute callback so the snapped value is the
    # one used to redraw the curve. g has two: the special g=1 and the fitted g.
    g_magnets = [1.0, round(g, 2)]
    magnet(tau_slider, [mtau], 0.5)
    magnet(xi_slider, [mxi], 0.1)
    magnet(g_slider, g_magnets, 0.2)

    # Gray dots on the track marking each magnet (default values). Bokeh 3 renders
    # the slider inside a shadow root, so we inject CSS *into* it via stylesheets;
    # the noUi track elements are reachable there. Up to two dots (::before/::after).
    def magnet_dots(slider, magnets, lo, hi):
        dot = ("content:'';position:absolute;top:50%;width:9px;height:9px;"
               "margin:-4.5px 0 0 -4.5px;border-radius:50%;background:#9e9e9e;"
               "pointer-events:none;z-index:5;")
        css = ".noUi-target{position:relative;}"
        for m, ps in zip(magnets, ("after", "before")):
            css += ".noUi-target::%s{%sleft:%.3f%%;}" % (ps, dot, (m - lo) / (hi - lo) * 100)
        slider.stylesheets = [InlineStyleSheet(css=css)]

    magnet_dots(tau_slider, [mtau], t_min, t_max)
    magnet_dots(xi_slider, [mxi], -1.5, 2.5)
    magnet_dots(g_slider, g_magnets, 0.2, 12.0)

    # One callback recomputes every curve. The main patient (index 0) follows the
    # τ/ξ sliders; the extras keep their fixed fitted (τ, ξ). All curves and the
    # population line share g, v0, so g/v0 reshape everything at once. The real
    # visit dots are data, not model -> they never move.
    callback = CustomJS(
        args=dict(srcs=curve_srcs, ptau=fixed_tau, pxi=fixed_xi, msrc=mean_src,
                  onset=onset, alpha_label=alpha_label, mid_line=mid_line,
                  mid_div=mid_div, tau=tau_slider, xi=xi_slider,
                  g=g_slider, v0=v0_slider, t0=t0),
        code="""
        const G = g.value, V0 = v0.value, metric = Math.pow(G + 1, 2) / G;
        for (let k = 0; k < srcs.length; k++) {
            const s = srcs[k], t = s.data['t'], y = s.data['y'];
            const TAU = (k === 0) ? tau.value : ptau[k];
            const XI  = (k === 0) ? xi.value  : pxi[k];
            const a = Math.exp(XI);
            for (let i = 0; i < t.length; i++)
                y[i] = 1 / (1 + G * Math.exp(-metric * V0 * a * (t[i] - TAU)));
            s.change.emit();
        }
        const mt = msrc.data['t'], my = msrc.data['y'];
        for (let i = 0; i < mt.length; i++)
            my[i] = 1 / (1 + G * Math.exp(-metric * V0 * (mt[i] - t0)));
        msrc.change.emit();
        onset.location = tau.value;
        alpha_label.text = "α = exp(ξ) = " + Math.exp(xi.value).toFixed(2);
        const showMid = Math.abs(g.value - 1) < 0.06;
        mid_line.visible = showMid;
        mid_div.text = showMid
            ? "<b>g = 1</b> → p(τ) = 0.5 : at the onset τ the trajectory is exactly at its midpoint."
            : "";
        """,
    )
    extra = ([g_slider] if show_g else []) + ([v0_slider] if show_v0 else [])
    sliders = [tau_slider, xi_slider] + extra
    for s in sliders:
        s.js_on_change("value", callback)

    return file_html(column(p, mid_div, *sliders), CDN, "leaspy interactive sigmoid")


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


def main():
    facts = model_facts()
    assets = Path(__file__).resolve().parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "sigmoid_interactive.html"
    out.write_text(build(facts, show_g=True), encoding="utf-8")  # sliders: τ, ξ, g
    tag = f"{facts['source']} data"
    if facts["source"] == "real":
        ids = ", ".join(f"{pt['role']}={pt['id']}" for pt in facts["patients"])
        tag += f": {facts['feature']}, g={facts['g']:.2f} v0={facts['v0']:.4f} " \
               f"tau_mean={facts['t0']:.1f} | {ids}"
    print(f"wrote {out}  ({tag})")

    # Reparametrization morph (raw spaghetti → reparametrized, animated).
    if facts["source"] == "real" and facts.get("all_patients"):
        out2 = assets / "reparam_morph.html"
        out2.write_text(build_morph(facts), encoding="utf-8")
        print(f"wrote {out2}  (highlighted: {', '.join(facts['highlight_ids'])})")


if __name__ == "__main__":
    main()
