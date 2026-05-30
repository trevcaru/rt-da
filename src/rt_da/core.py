"""Core unequal-variance SDT fitting (RT-based da).

Direct port of the reference implementation (Miyoshi et al. 2026,
kiyomiyoshi/rt_type1_roc: uvsdt.R), validated to reproduce its published
example exactly and its per-subject estimates on real datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class SDTFit:
    """Result of an unequal-variance SDT fit.

    Attributes
    ----------
    mu : float
        Mean of the target-present distribution (target-absent fixed at 0).
    sigma : float
        SD ratio = SD_present / SD_absent. >1 means present is more variable.
    da : float
        Unequal-variance sensitivity, mu / sqrt((1 + sigma**2) / 2).
    dprime : float
        Conventional equal-variance d' from the single yes/no midpoint
        (z(hit) - z(fa)), for comparison.
    criterion : float
        Conventional criterion c = -(z(hit) + z(fa)) / 2.
    criteria : np.ndarray
        The 2*n_bins - 1 fitted decision criteria along the evidence axis.
    log_likelihood : float
        Maximized log-likelihood of the fitted model.
    n_bins : int
        Number of rating bins used.
    converged : bool
        Whether the optimizer reported success.
    valid : bool
        Whether the fit is usable: converged AND not pinned to a parameter
        boundary (e.g. sigma not maxed out). Degenerate fits get valid=False
        so they can be excluded, mirroring the reference R pipeline's
        convergence-based subject exclusions.
    n_trials : int
        Number of trials that entered the fit.
    """
    mu: float
    sigma: float
    da: float
    dprime: float
    criterion: float
    criteria: np.ndarray = field(repr=False)
    log_likelihood: float
    n_bins: int
    converged: bool
    n_trials: int
    valid: bool = True

    def as_dict(self) -> dict:
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "da": self.da,
            "dprime": self.dprime,
            "criterion": self.criterion,
            "log_likelihood": self.log_likelihood,
            "n_bins": self.n_bins,
            "converged": self.converged,
            "valid": self.valid,
            "n_trials": self.n_trials,
        }


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _as_int_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    return arr


def rt_to_bins(rt: Sequence[float],
               response: Sequence[int] = None,
               n_bins: int = 3,
               per_response: bool = False) -> np.ndarray:
    """Convert response times to confidence-like rating bins.

    The FASTEST responses get the HIGHEST rating (n_bins), matching the
    assumption that faster = more confident (Miyoshi et al.).

    By default this matches the paper's procedure (Figure 2a): RTs are split
    into `n_bins` equal-sized quantile bins with stimulus class and response
    COLLAPSED -- i.e. one common set of tertile cutoffs across all trials.
    Set `per_response=True` to instead bin within each response type
    separately (useful when yes/no RT distributions differ markedly).

    Parameters
    ----------
    rt : array-like of float
        Response times (any positive unit). NaN / non-positive -> unrated.
    response : array-like of int, optional
        1 = "yes", 0/2 = "no". Only needed if `per_response=True`.
    n_bins : int
        Number of quantile bins.
    per_response : bool
        If True, bin within each response type separately. Default False
        (paper-faithful: collapsed binning).

    Returns
    -------
    np.ndarray of int
        Ratings in 1..n_bins, or 0 where the trial could not be binned.
    """
    rt = np.asarray(rt, dtype=float)
    ratings = np.zeros(len(rt), dtype=int)

    def _bin(idx):
        if idx.size == 0:
            return
        if idx.size < n_bins:
            ratings[idx] = int(np.ceil(n_bins / 2))
            return
        try:
            # Rank by -rt so the fastest RT lands in the top quantile.
            q = pd.qcut(-rt[idx], n_bins, labels=False, duplicates="drop")
        except ValueError:
            ratings[idx] = int(np.ceil(n_bins / 2))
            return
        ratings[idx] = q.astype(int) + 1

    if per_response:
        if response is None:
            raise ValueError("response required when per_response=True")
        is_yes = (np.asarray(response) == 1)
        for yes in (True, False):
            _bin(np.flatnonzero((is_yes == yes) & np.isfinite(rt) & (rt > 0)))
    else:
        # Paper-faithful: collapsed across stimulus and response.
        _bin(np.flatnonzero(np.isfinite(rt) & (rt > 0)))
    return ratings


def build_roc_table(stimulus: Sequence[int],
                    response: Sequence[int],
                    rating: Sequence[int],
                    n_bins: int = 3,
                    stabilize: Optional[float] = None):
    """Build the 2 x 2n response-frequency table used to fit the UV model.

    Cell ordering follows Miyoshi et al.'s `nr_s1`/`nr_s2` convention exactly
    (their README / Figure 2b): cells run left-to-right from
    "fastest-RT / highest-confidence YES" to "fastest-RT / highest-confidence
    NO" -- i.e. DECREASING support for a "yes" judgement.

        cell 1            .. n_bins        : "yes" responses, rating high..low
        cell n_bins+1     .. 2*n_bins      : "no"  responses, rating low..high

    So `nr_present` (their nr_s2) is front-loaded and `nr_absent` (nr_s1)
    is back-loaded, matching their example
    nr_s2 = c(43,21,10,12,8,3), nr_s1 = c(10,7,16,27,29,10).

    Parameters
    ----------
    stimulus : array-like of int
        1 = target present, 0/2 = target absent.
    response : array-like of int
        1 = "yes", 0/2 = "no".
    rating : array-like of int
        Rating bin 1..n_bins (e.g. from `rt_to_bins`). 0 = exclude trial.
    n_bins : int
        Number of rating levels.
    stabilize : float, optional
        Constant added to every cell (their `add_constant`). The paper /
        their code default is 1 / total_trials. If None, that is applied.

    Returns
    -------
    (nr_absent, nr_present) : tuple of np.ndarray   (their nr_s1, nr_s2)
        Each length 2*n_bins.
    """
    stimulus = np.asarray(stimulus)
    response = np.asarray(response)
    rating = np.asarray(rating)

    present = (stimulus == 1)
    yes = (response == 1)
    valid = (rating >= 1) & (rating <= n_bins)

    # Paper ordering: highest support for "yes" on the LEFT.
    #   "yes": cell = n_bins - rating + 1   (fastest/most-confident yes -> 1)
    #   "no" : cell = n_bins + rating       (fastest/most-confident no  -> 2n)
    cell = np.where(yes, n_bins - rating + 1, n_bins + rating)

    all_cells = np.arange(1, 2 * n_bins + 1)

    def counts(sel):
        sub = cell[sel & valid]
        return np.array([(sub == c).sum() for c in all_cells], dtype=float)

    nr_present = counts(present)   # nr_s2
    nr_absent = counts(~present)   # nr_s1

    if stabilize is None:
        # Match the reference `add_constant`: 1 / (number of cells) = 1/(2n).
        stabilize = 1.0 / (2 * n_bins)
    nr_present = nr_present + stabilize
    nr_absent = nr_absent + stabilize

    return nr_absent, nr_present


# ---------------------------------------------------------------------------
# The unequal-variance SDT likelihood + fit.
# This is a direct port of the authors' reference implementation (uvsdt.R:
# uvsdt_logL + fit_uvsdt_mle), so per-subject estimates match theirs.
# ---------------------------------------------------------------------------

def _neg_log_likelihood(params: np.ndarray,
                        nr_s1: np.ndarray,
                        nr_s2: np.ndarray,
                        n_ratings: int) -> float:
    """Negative log-likelihood, ported verbatim from uvsdt_logL (uvsdt.R).

    params = [mu, sigma, cri_1, ..., cri_{2n-1}].
    Target-absent (S1) ~ N(0, 1); target-present (S2) ~ N(mu, sigma).
    Predicted cumulative rates use the authors' criterion convention:
        pred_far = [0, Phi(0 - cri),            1]
        pred_hr  = [0, Phi((mu - cri)/sigma),   1]
    Cell probabilities are the successive diffs. If any diff is <= 0 the
    log is NaN, which (as in the R code) is mapped to +inf (worst score),
    so the optimizer simply avoids those regions -- no sorting or penalty
    term is needed.
    """
    mu = params[0]
    sigma = params[1]
    cri = params[2:2 * n_ratings + 1]

    if sigma <= 0:
        return np.inf

    pred_far = np.concatenate(([0.0], norm.cdf(0.0 - cri), [1.0]))
    pred_hr = np.concatenate(([0.0], norm.cdf((mu - cri) / sigma), [1.0]))

    pred_nr_s1 = nr_s1.sum() * np.diff(pred_far)
    pred_nr_s2 = nr_s2.sum() * np.diff(pred_hr)

    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.sum(nr_s1 * np.log(pred_nr_s1 / nr_s1.sum())) + \
            np.sum(nr_s2 * np.log(pred_nr_s2 / nr_s2.sum()))
    if not np.isfinite(ll):
        return np.inf
    return -ll


def fit_uv_sdt(nr_absent: Sequence[float],
               nr_present: Sequence[float],
               n_bins: Optional[int] = None) -> SDTFit:
    """Fit the unequal-variance SDT model by maximum likelihood.

    Parameters
    ----------
    nr_absent, nr_present : array-like of float
        Response-frequency counts (length 2*n_bins), e.g. from
        `build_roc_table`. Should already be stabilized (no zero cells).
    n_bins : int, optional
        Number of rating bins; inferred from array length if omitted.

    Returns
    -------
    SDTFit
    """
    nr_s1 = np.asarray(nr_absent, dtype=float)   # S1 = target-absent
    nr_s2 = np.asarray(nr_present, dtype=float)  # S2 = target-present
    if nr_s1.shape != nr_s2.shape:
        raise ValueError("nr_absent and nr_present must have the same length")
    if n_bins is None:
        n_bins = nr_s1.size // 2
    n_ratings = n_bins

    # Initial guess, exactly as in uvsdt.R:
    #   rating_far = cumsum(nr_s1)/sum(nr_s1);  rating_hr = cumsum(nr_s2)/sum
    #   mu  = qnorm(rating_hr[n]) - qnorm(rating_far[n])
    #   sigma = 1.5
    #   cri = -qnorm(rating_far)[1:(2n-1)]
    rating_far = np.cumsum(nr_s1) / nr_s1.sum()
    rating_hr = np.cumsum(nr_s2) / nr_s2.sum()
    # clip only to keep qnorm finite at the seed (does not affect the fit)
    rf = np.clip(rating_far, 1e-6, 1 - 1e-6)
    rh = np.clip(rating_hr, 1e-6, 1 - 1e-6)
    mu0 = norm.ppf(rh[n_ratings - 1]) - norm.ppf(rf[n_ratings - 1])
    sigma0 = 1.5
    cri0 = (-norm.ppf(rf))[:2 * n_ratings - 1]
    guess = np.concatenate(([mu0, sigma0], cri0))

    # Fit with BFGS, matching the reference (suppressWarnings(optim(...,
    # method = "BFGS", maxit = 10000))).
    res = minimize(
        _neg_log_likelihood, guess,
        args=(nr_s1, nr_s2, n_ratings),
        method="BFGS",
        options={"maxiter": 10000},
    )

    mu, sigma = float(res.x[0]), float(res.x[1])
    da = mu / np.sqrt((1.0 + sigma ** 2) / 2.0)

    # Criteria, output in the reference's reversed order. In R (1-indexed):
    #   cri[i] = fit$par[2n + 2 - i]  for i = 1..2n-1  -> par indices 2n+1..3
    # In 0-indexed Python that is par[2n] down to par[2].
    par = res.x
    criteria = np.array([par[2 * n_ratings - i]
                         for i in range(2 * n_ratings - 1)])

    # Validity: the reference keeps any fit that did not error and is finite
    # (it relies on na.omit + d'>=0 downstream rather than parameter ceilings).
    valid = bool(np.isfinite(mu) and np.isfinite(sigma) and sigma > 0
                 and np.isfinite(da) and np.all(np.isfinite(criteria)))

    # Conventional equal-variance d' from the single yes/no midpoint.
    # Paper ordering: cells 1..n are "yes", n+1..2n are "no".
    mid = n_ratings
    hit = np.clip(nr_s2[:mid].sum() / nr_s2.sum(), 1e-6, 1 - 1e-6)
    fa = np.clip(nr_s1[:mid].sum() / nr_s1.sum(), 1e-6, 1 - 1e-6)
    z_hit, z_fa = norm.ppf(hit), norm.ppf(fa)
    dprime = z_hit - z_fa
    criterion = -0.5 * (z_hit + z_fa)

    return SDTFit(
        mu=mu,
        sigma=sigma,
        da=float(da),
        dprime=float(dprime),
        criterion=float(criterion),
        criteria=criteria,
        log_likelihood=float(-res.fun),
        n_bins=int(n_bins),
        converged=bool(res.success),
        n_trials=int(round(nr_s1.sum() + nr_s2.sum())),
        valid=valid,
    )


def fit_uvsdt_mle(nr_s1, nr_s2, add_constant: bool = True) -> SDTFit:
    """Drop-in match for the authors' R function `fit_uvsdt_mle()`.

    Mirrors the signature and conventions from the reference repository
    (kiyomiyoshi/rt_type1_roc): `nr_s1` and `nr_s2` are response-frequency
    vectors for S1 (target-absent) and S2 (target-present) trials, ordered
    from "fastest-RT / highest-confidence Yes" to "fastest-RT / highest-
    confidence No". `add_constant=True` adds 1/sum(counts) to each cell for
    stability (their default).

    Returns an SDTFit with mu, sigma, da, the criteria (cri.X1..), and logL,
    so results can be checked directly against the paper's example.
    """
    nr_s1 = np.asarray(nr_s1, dtype=float)
    nr_s2 = np.asarray(nr_s2, dtype=float)
    if add_constant:
        # The reference implementation adds 1 / (number of response
        # categories) to each cell, i.e. 1/(2n). (The paper's prose says
        # "1/total trials", but their code -- and the published example
        # numbers -- use 1/len. We match the code.)
        const = 1.0 / nr_s1.size
        nr_s1 = nr_s1 + const
        nr_s2 = nr_s2 + const
    n_bins = nr_s1.size // 2
    return fit_uv_sdt(nr_s1, nr_s2, n_bins=n_bins)


# ---------------------------------------------------------------------------
# Top-level convenience wrappers
# ---------------------------------------------------------------------------

def fit_ratings(stimulus, response, rating, n_bins=3,
                stabilize=None) -> SDTFit:
    """Fit da from explicit ratings (RT bins OR real confidence).

    This is the rating-agnostic entry point. Pass confidence ratings to
    reproduce the paper's confidence-based da; pass RT quantile bins (from
    `rt_to_bins`) for the RT-based da.
    """
    nr_absent, nr_present = build_roc_table(
        stimulus, response, rating, n_bins=n_bins, stabilize=stabilize)
    return fit_uv_sdt(nr_absent, nr_present, n_bins=n_bins)


def rt_da(stimulus, response, rt, n_bins=3, stabilize=None) -> SDTFit:
    """Estimate RT-based da directly from trial-level stimulus/response/RT.

    Parameters
    ----------
    stimulus : array-like, 1 = present, 0/2 = absent
    response : array-like, 1 = "yes", 0/2 = "no"
    rt : array-like of float, response times (positive)
    n_bins : int, number of RT quantile bins (match your confidence scale)
    stabilize : float, optional cell constant (default 1 / n_trials)
    """
    rating = rt_to_bins(rt, response, n_bins=n_bins)
    return fit_ratings(stimulus, response, rating, n_bins=n_bins,
                       stabilize=stabilize)


