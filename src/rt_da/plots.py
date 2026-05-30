"""Plotting (matplotlib imported lazily; install via the [plots] extra).

All functions accept an existing Axes or create one. The module-level
helpers build the standard diagnostic figures used in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .core import SDTFit, fit_uv_sdt, rt_to_bins, build_roc_table, fit_ratings
from .simulate import simulate_detection
from .group import fit_group



# A small, consistent visual identity for all figures.
_PALETTE = {
    "absent":   "#D1495B",   # target-absent / noise   (warm red)
    "present":  "#2E5EAA",   # target-present / signal (deep blue)
    "rt":       "#E08E45",   # RT-based                 (amber)
    "conf":     "#2E8B8B",   # confidence-based         (teal)
    "accent":   "#1B1B3A",   # ink / model lines
    "muted":    "#9AA0A6",   # reference / unity lines
    "grid":     "#E6E8EB",
    "fill":     "#EEF2F7",
}


def _style_ax(ax, title=None, xlabel=None, ylabel=None):
    """Apply the shared clean look to an Axes (spines, grid, fonts)."""
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=_PALETTE["accent"], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, color="#33373B")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, color="#33373B")
    ax.tick_params(labelsize=9, colors="#5F6368", length=0)
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C6CACE")
        ax.spines[side].set_linewidth(1.0)
    return ax


def _roc_points(nr_absent, nr_present, n_bins):
    """Hit/FA rates across progressively lenient criteria.

    Cells are high-evidence-first (paper ordering), so cumulate from the
    LEFT to sweep from the strictest to the most lenient criterion.
    """
    tot_p, tot_a = nr_present.sum(), nr_absent.sum()
    hr, far = [0.0], [0.0]  # strictest point (0,0)
    for c in range(1, 2 * n_bins):
        hr.append(nr_present[:c].sum() / tot_p)
        far.append(nr_absent[:c].sum() / tot_a)
    hr.append(1.0); far.append(1.0)  # most lenient point (1,1)
    return np.array(far), np.array(hr)


def plot_roc(fit: SDTFit, nr_absent, nr_present, ax=None):
    """Plot the empirical type-1 ROC with the fitted UV model curve."""
    import matplotlib.pyplot as plt  # lazy import
    nr_absent = np.asarray(nr_absent, float)
    nr_present = np.asarray(nr_present, float)
    far, hr = _roc_points(nr_absent, nr_present, fit.n_bins)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.6))

    # chance diagonal
    ax.plot([0, 1], [0, 1], "--", color=_PALETTE["muted"], lw=1.2,
            zorder=1, label="chance")

    # fitted model ROC (smooth curve via continuous criterion sweep)
    cc = np.linspace(-6, 6, 500)
    model_far = 1 - norm.cdf(cc, 0, 1)
    model_hr = 1 - norm.cdf(cc, fit.mu, fit.sigma)
    ax.plot(model_far, model_hr, "-", color=_PALETTE["present"], lw=2.4,
            zorder=2, label=f"UV fit  (da={fit.da:.2f}, σ={fit.sigma:.2f})")
    ax.fill_between(model_far, model_hr, model_far,
                    color=_PALETTE["present"], alpha=0.07, zorder=1)

    # empirical points
    ax.scatter(far, hr, s=70, facecolor="white",
               edgecolor=_PALETTE["accent"], linewidth=1.8, zorder=4,
               label="empirical")

    _style_ax(ax, xlabel="False-alarm rate", ylabel="Hit rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    return ax


def plot_zroc(fit: SDTFit, nr_absent, nr_present, ax=None):
    """Plot the z-transformed ROC; slope = 1/sigma."""
    import matplotlib.pyplot as plt  # lazy import
    nr_absent = np.asarray(nr_absent, float)
    nr_present = np.asarray(nr_present, float)
    far, hr = _roc_points(nr_absent, nr_present, fit.n_bins)
    interior = (far > 0) & (far < 1) & (hr > 0) & (hr < 1)
    zf, zh = norm.ppf(far[interior]), norm.ppf(hr[interior])

    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.6))

    xs = np.linspace(zf.min() - 0.6, zf.max() + 0.6, 100)
    # equal-variance reference (slope 1)
    ax.plot(xs, xs + (fit.mu), "--", color=_PALETTE["muted"], lw=1.2,
            zorder=1, label="equal-var (slope 1)")
    # fitted line: z(hit) = (1/sigma) z(fa) + mu/sigma
    ax.plot(xs, xs / fit.sigma + fit.mu / fit.sigma, "-",
            color=_PALETTE["present"], lw=2.4, zorder=2,
            label=f"UV fit  (slope 1/σ = {1/fit.sigma:.2f})")
    ax.scatter(zf, zh, s=70, facecolor="white",
               edgecolor=_PALETTE["accent"], linewidth=1.8, zorder=4,
               label="empirical")

    _style_ax(ax, xlabel="z(FA rate)", ylabel="z(hit rate)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    return ax


def plot_distributions(fit: SDTFit, ax=None):
    """Plot the two internal SDT distributions with the decision criterion.

    Mirrors the paper's Figure 1C: target-absent ~ N(0,1) and target-present
    ~ N(mu, sigma), with the main yes/no criterion marked. The visual makes
    the unequal variance (and resulting ROC asymmetry) immediately legible.
    """
    import matplotlib.pyplot as plt  # lazy import
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.0))

    lo = min(-3.5, fit.mu - 3.5 * fit.sigma)
    hi = max(3.5, fit.mu + 3.5 * fit.sigma)
    x = np.linspace(lo, hi, 600)
    y_absent = norm.pdf(x, 0, 1)
    y_present = norm.pdf(x, fit.mu, fit.sigma)

    ax.fill_between(x, y_absent, color=_PALETTE["absent"], alpha=0.30,
                    zorder=2)
    ax.fill_between(x, y_present, color=_PALETTE["present"], alpha=0.25,
                    zorder=2)
    ax.plot(x, y_absent, color=_PALETTE["absent"], lw=2.2, zorder=3,
            label="target absent  N(0, 1)")
    ax.plot(x, y_present, color=_PALETTE["present"], lw=2.2, zorder=3,
            label=f"target present  N({fit.mu:.2f}, {fit.sigma:.2f})")

    # main decision criterion = the middle fitted criterion (yes/no boundary)
    crit = fit.criteria[len(fit.criteria) // 2]
    ax.axvline(crit, color=_PALETTE["accent"], lw=1.6, ls=(0, (4, 2)),
               zorder=4)
    # label placed low and vertically along the line, clear of the legend
    # (upper-right) and the da box (upper-left)
    ax.text(crit, ax.get_ylim()[1] * 0.45, "criterion ", rotation=90,
            color=_PALETTE["accent"], fontsize=8.5, va="center", ha="right")

    # annotate da
    ax.text(0.02, 0.96, f"da = {fit.da:.2f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=_PALETTE["accent"],
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc=_PALETTE["fill"],
                      ec="none"))

    _style_ax(ax, xlabel="Internal signal strength", ylabel="Density")
    ax.set_yticks([])
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    return ax


def plot_da_scatter(da_rt, da_conf, ax=None):
    """Scatter da(RT) vs da(Conf) across subjects (the paper's Figure 9)."""
    import matplotlib.pyplot as plt  # lazy import
    da_rt = np.asarray(da_rt, float)
    da_conf = np.asarray(da_conf, float)
    ok = np.isfinite(da_rt) & np.isfinite(da_conf)
    da_rt, da_conf = da_rt[ok], da_conf[ok]

    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.8))

    lo = min(da_rt.min(), da_conf.min())
    hi = max(da_rt.max(), da_conf.max())
    pad = 0.12 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--",
            color=_PALETTE["muted"], lw=1.2, zorder=1, label="unity")

    if da_rt.size >= 2 and not np.allclose(da_conf, da_conf[0]):
        b, a = np.polyfit(da_conf, da_rt, 1)
        xs = np.array([lo - pad, hi + pad])
        ax.plot(xs, a + b * xs, "-", color=_PALETTE["accent"], lw=2.0,
                zorder=2, label=f"fit: y = {b:.2f}x + {a:.2f}")
        r = np.corrcoef(da_conf, da_rt)[0, 1]
        title = f"da agreement   (r = {r:.2f})"
    else:
        title = "da agreement"

    ax.scatter(da_conf, da_rt, s=55, facecolor=_PALETTE["conf"],
               edgecolor="white", linewidth=0.8, alpha=0.85, zorder=3)

    _style_ax(ax, title=title, xlabel="confidence-based da",
              ylabel="RT-based da")
    ax.set_aspect("equal")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    return ax


def plot_recovery(true_da, est_da, ax=None, label="da"):
    """Scatter true vs estimated parameter (parameter-recovery check)."""
    import matplotlib.pyplot as plt  # lazy import
    true_da = np.asarray(true_da, float)
    est_da = np.asarray(est_da, float)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.8))

    lo = min(true_da.min(), est_da.min())
    hi = max(true_da.max(), est_da.max())
    pad = 0.12 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--",
            color=_PALETTE["muted"], lw=1.2, zorder=1,
            label="perfect recovery")
    ax.scatter(true_da, est_da, s=55, facecolor=_PALETTE["rt"],
               edgecolor="white", linewidth=0.8, alpha=0.85, zorder=3)

    _style_ax(ax, title=f"{label} recovery", xlabel=f"true {label}",
              ylabel=f"estimated {label}")
    ax.set_aspect("equal")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    return ax


def plot_overestimation(grp_df, ax=None):
    """Grouped bar plot of d' vs da(RT) vs da(Conf) (paper Figures 3 & 6).

    Shows that conventional d' overestimates sensitivity relative to both
    unequal-variance da measures. Bars are dataset means with SEM error bars.
    """
    import matplotlib.pyplot as plt  # lazy import
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2))

    def msem(col):
        v = np.asarray(grp_df[col], float)
        v = v[np.isfinite(v)]
        return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0)

    labels = ["d′\n(equal-var)", "da (RT)", "da (Conf)"]
    cols = ["dprime_rt", "da_rt", "da_conf"]
    colors = [_PALETTE["accent"], _PALETTE["rt"], _PALETTE["conf"]]
    means, sems = zip(*[msem(c) for c in cols])

    xpos = np.arange(len(labels))
    bars = ax.bar(xpos, means, yerr=sems, width=0.62, color=colors,
                  edgecolor="white", linewidth=1.2, zorder=3,
                  error_kw=dict(ecolor="#5F6368", lw=1.3, capsize=4))
    # place each value label just above the top of its error bar so the
    # number never collides with the error-bar cap
    for b, m, e in zip(bars, means, sems):
        ax.text(b.get_x() + b.get_width() / 2, m + e + max(means) * 0.03,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=_PALETTE["accent"])

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=9.5)
    _style_ax(ax, title="d′ overestimates sensitivity vs da",
              ylabel="sensitivity")
    ax.set_ylim(0, max(means) * 1.30)
    ax.grid(axis="x", visible=False)
    return ax


def _make_plots(save_dir=None, show=True):
    """Generate, save, and (optionally) open the standard diagnostic figures.

    Figures are written next to this script by default so they are easy to
    find regardless of where the script was launched from.
    """
    import os
    import subprocess
    import matplotlib.pyplot as plt  # backend already set to Agg at top
    _interactive = False

    # global typographic defaults for a cleaner look
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 110,
    })

    if save_dir is None:
        try:
            save_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            save_dir = os.getcwd()

    paths = []

    # ---- Figure 1: model overview — distributions + ROC + z-ROC ----------
    df = simulate_detection(n_trials=4000, mu=1.5, sigma=1.5, seed=3)
    rating = rt_to_bins(df.rt, df.response, n_bins=3)
    na, npz = build_roc_table(df.stimulus, df.response, rating, n_bins=3)
    fit = fit_uv_sdt(na, npz, n_bins=3)

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.28,
                          left=0.06, right=0.97, bottom=0.14, top=0.84)
    plot_distributions(fit, ax=fig.add_subplot(gs[0]))
    plot_roc(fit, na, npz, ax=fig.add_subplot(gs[1]))
    plot_zroc(fit, na, npz, ax=fig.add_subplot(gs[2]))
    fig.suptitle("RT-based unequal-variance SDT — model overview   "
                 f"(true da = 1.18, recovered da = {fit.da:.2f}, "
                 f"σ = {fit.sigma:.2f})",
                 fontsize=13, fontweight="bold", color=_PALETTE["accent"],
                 y=0.97)
    p = os.path.join(save_dir, "rtda_fig1_model_overview.png")
    fig.savefig(p, dpi=130); paths.append(p)

    # ---- group fit reused by figures 2 & 4 -------------------------------
    frames = []
    for subj in range(30):
        d = simulate_detection(n_trials=800, mu=0.8 + 0.05 * subj,
                               sigma=1.4, seed=subj)
        d["subject"] = subj
        frames.append(d)
    big = pd.concat(frames, ignore_index=True)
    grp = fit_group(big, subject="subject", stimulus="stimulus",
                    response="response", rt="rt", confidence="confidence",
                    n_bins=3)

    # ---- Figure 2: RT vs confidence — scatter + overestimation bars ------
    fig2 = plt.figure(figsize=(10.5, 4.7))
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.28,
                            left=0.08, right=0.96, bottom=0.14, top=0.86)
    plot_da_scatter(grp["da_rt"], grp["da_conf"], ax=fig2.add_subplot(gs2[0]))
    plot_overestimation(grp, ax=fig2.add_subplot(gs2[1]))
    fig2.suptitle("RT vs confidence agreement, and d′ overestimation",
                  fontsize=13, fontweight="bold", color=_PALETTE["accent"],
                  y=0.97)
    p = os.path.join(save_dir, "rtda_fig2_rt_vs_confidence.png")
    fig2.savefig(p, dpi=130); paths.append(p)

    # ---- Figure 3: parameter recovery ------------------------------------
    truths, ests = [], []
    for i, (tmu, tsig) in enumerate([(1.0, 1.2), (1.5, 1.3), (1.5, 1.6),
                                     (2.0, 1.5), (2.2, 1.8), (0.8, 1.1)]):
        tda = tmu / np.sqrt((1 + tsig ** 2) / 2)
        for s in range(5):
            df = simulate_detection(n_trials=2000, mu=tmu, sigma=tsig,
                                    seed=100 * i + s)
            f = fit_ratings(df.stimulus, df.response, df.confidence, n_bins=3)
            truths.append(tda); ests.append(f.da)
    fig3, ax = plt.subplots(figsize=(5.0, 5.0))
    plot_recovery(truths, ests, ax=ax, label="da")
    fig3.tight_layout()
    p = os.path.join(save_dir, "rtda_fig3_recovery.png")
    fig3.savefig(p, dpi=130); paths.append(p)

    if show:
        _show_in_windows(paths)

    return paths


def _show_in_windows(paths):
    """Open the saved figures in real matplotlib windows (not Paint).

    We built the figures with the headless Agg backend for a fast, reliable
    import. To display them as interactive windows we switch to a GUI backend
    and render each saved PNG into its own matplotlib window. If no GUI
    backend is available (e.g. a bare server), we fall back to handing the
    files to the OS image viewer.
    """
    import os
    import subprocess
    import matplotlib
    # Try to switch to a GUI backend now that the heavy lifting is done.
    gui_ok = False
    for backend in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX", "WxAgg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt
            gui_ok = True
            break
        except Exception:
            continue

    if gui_ok:
        try:
            import matplotlib.pyplot as plt
            for p in paths:
                img = plt.imread(p)
                # size the window to the image's aspect ratio
                h, w = img.shape[0], img.shape[1]
                fig = plt.figure(figsize=(min(w / 130, 13), min(h / 130, 9)))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(img)
                ax.axis("off")
                fig.canvas.manager.set_window_title(os.path.basename(p))
            print("\nOpening figures in matplotlib windows. "
                  "Close the windows to finish.", flush=True)
            plt.show()  # blocks until the user closes the windows
            return
        except Exception as e:
            print(f"(GUI display failed: {e}; falling back to image viewer.)",
                  flush=True)

    # Fallback: no GUI backend -> open saved PNGs with the OS default app.
    for p in paths:
        try:
            if sys.platform.startswith("win"):
                os.startfile(p)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception:
            pass





