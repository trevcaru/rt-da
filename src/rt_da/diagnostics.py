"""Diagnostics: RT-validity check, bootstrap CIs, and the dual
RT-vs-confidence reporting that is the package's signature output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

from .core import SDTFit, rt_da, fit_ratings


@dataclass
class RTValidity:
    rho: float
    p_value: float
    n: int
    verdict: str

    def __repr__(self):
        return (f"RTValidity(rho={self.rho:.3f}, p={self.p_value:.3g}, "
                f"n={self.n}, verdict={self.verdict!r})")


def rt_validity(rt, confidence) -> RTValidity:
    """Check whether RT actually proxies confidence in THIS data.

    The whole method rests on faster RT => higher confidence. Miyoshi et al.
    explicitly warn it fails in some domains (e.g. memory tasks). If you have
    both RT and confidence on the same trials, this returns the Spearman
    correlation (expected NEGATIVE: faster -> higher confidence) plus a plain
    verdict. Run this before trusting RT-based da.
    """
    rt = np.asarray(rt, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    mask = np.isfinite(rt) & np.isfinite(confidence) & (rt > 0)
    rt, confidence = rt[mask], confidence[mask]
    n = rt.size
    if n < 10:
        return RTValidity(np.nan, np.nan, n, "insufficient data (n<10)")
    rho, p = spearmanr(rt, confidence)
    if rho < 0 and p < 0.05:
        verdict = "valid: faster RT predicts higher confidence (as expected)"
    elif rho < 0:
        verdict = "weak: negative trend but not significant"
    else:
        verdict = "INVALID: RT does not track confidence here; da unreliable"
    return RTValidity(float(rho), float(p), int(n), verdict)


# ---------------------------------------------------------------------------
# Flagship feature 2: bootstrap confidence intervals
# ---------------------------------------------------------------------------

def rt_da_ci(stimulus, response, rt, n_bins=3, stabilize=None,
             n_boot=2000, ci=0.95, seed=None, rating=None):
    """Bootstrap confidence intervals for da, sigma, mu.

    Resamples trials with replacement, refits each time, and returns
    percentile CIs. Pass `rating=` (e.g. confidence) to bootstrap a
    rating-based fit instead of RT.

    Returns
    -------
    dict mapping 'da'/'sigma'/'mu' -> (point_estimate, lo, hi), plus the
    full SDTFit on the original data under key 'fit'.
    """
    rng = np.random.default_rng(seed)
    stimulus = np.asarray(stimulus)
    response = np.asarray(response)
    n = len(stimulus)

    if rating is None:
        rt = np.asarray(rt, dtype=float)
        base = rt_da(stimulus, response, rt, n_bins=n_bins, stabilize=stabilize)
    else:
        rating = np.asarray(rating)
        base = fit_ratings(stimulus, response, rating, n_bins=n_bins,
                           stabilize=stabilize)

    das, sigmas, mus = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            if rating is None:
                f = rt_da(stimulus[idx], response[idx], rt[idx],
                          n_bins=n_bins, stabilize=stabilize)
            else:
                f = fit_ratings(stimulus[idx], response[idx], rating[idx],
                               n_bins=n_bins, stabilize=stabilize)
        except Exception:
            continue
        if f.converged and np.isfinite(f.da):
            das.append(f.da); sigmas.append(f.sigma); mus.append(f.mu)

    alpha = (1 - ci) / 2
    def pct(arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return (np.nan, np.nan)
        return (float(np.quantile(arr, alpha)),
                float(np.quantile(arr, 1 - alpha)))

    return {
        "da": (base.da, *pct(das)),
        "sigma": (base.sigma, *pct(sigmas)),
        "mu": (base.mu, *pct(mus)),
        "n_boot_used": len(das),
        "fit": base,
    }


@dataclass
class DualEstimate:
    """RT-based and confidence-based da reported side by side.

    This is the package's signature output: it never hides the RT estimate
    behind the confidence one (or vice versa). It reports both, their
    agreement, and -- when both are available -- whether RT is behaving as a
    valid proxy in THIS data.
    """
    rt_fit: Optional[SDTFit]
    conf_fit: Optional[SDTFit]
    validity: Optional[RTValidity]

    @property
    def da_rt(self):
        return self.rt_fit.da if self.rt_fit else np.nan

    @property
    def da_conf(self):
        return self.conf_fit.da if self.conf_fit else np.nan

    @property
    def da_diff(self):
        """confidence-based da minus RT-based da (paper: RT runs ~5% lower)."""
        return self.da_conf - self.da_rt

    @property
    def da_pct_diff(self):
        if self.conf_fit and self.da_conf != 0:
            return 100.0 * self.da_diff / self.da_conf
        return np.nan

    def summary(self) -> str:
        lines = ["RT-based vs confidence-based unequal-variance SDT", "-" * 50]
        if self.rt_fit:
            lines.append(f"  RT   : da={self.rt_fit.da:.3f}  "
                         f"sigma={self.rt_fit.sigma:.3f}  "
                         f"mu={self.rt_fit.mu:.3f}  d'={self.rt_fit.dprime:.3f}")
        if self.conf_fit:
            lines.append(f"  Conf : da={self.conf_fit.da:.3f}  "
                         f"sigma={self.conf_fit.sigma:.3f}  "
                         f"mu={self.conf_fit.mu:.3f}  "
                         f"d'={self.conf_fit.dprime:.3f}")
        if self.rt_fit and self.conf_fit:
            lines.append(f"  da agreement: diff={self.da_diff:+.3f} "
                         f"({self.da_pct_diff:+.1f}% of confidence-based)")
            d = self.rt_fit
            lines.append(f"  d' overestimates da(RT) by "
                         f"{d.dprime - d.da:+.3f} (expected >0 when sigma>1)")
        if self.validity:
            lines.append(f"  RT validity: {self.validity.verdict} "
                         f"(rho={self.validity.rho:.3f})")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()


def compare_rt_confidence(stimulus, response, rt, confidence,
                          n_bins=3, stabilize=None) -> DualEstimate:
    """Fit the SAME UV-SDT model from RT and from confidence; report both.

    Reproduces the paper's core comparison at the single-subject level and
    is the recommended top-level call whenever both RT and confidence exist.
    """
    rt_fit = rt_da(stimulus, response, rt, n_bins=n_bins, stabilize=stabilize)
    conf_fit = fit_ratings(stimulus, response, confidence, n_bins=n_bins,
                           stabilize=stabilize)
    validity = rt_validity(rt, confidence)
    return DualEstimate(rt_fit=rt_fit, conf_fit=conf_fit, validity=validity)



