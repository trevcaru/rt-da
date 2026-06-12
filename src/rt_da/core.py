"""Core unequal-variance SDT fitting (RT-based da).

Independent reimplementation following the logic of the reference
(Miyoshi et al. 2026, kiyomiyoshi/rt_type1_roc: uvsdt.R). It reproduces the
authors' published example to the printed digits and matches their
per-subject mu/sigma/da/logL to optimizer precision against their own R --
see the `validation/` suite. The model is fit in a smooth, unconstrained
parameterization (see `_unpack`) so the optimizer is well-behaved.
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
        Whether the fit is usable: the optimizer converged (`converged`) AND
        the estimates pass sanity bounds (finite mu/da, 0.1 < sigma < 8,
        finite strictly-ordered criteria). Degenerate / non-converged fits get
        valid=False so they can be excluded, mirroring the reference R
        pipeline's na.omit-based subject exclusions.
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
        Constant added to every cell (their `add_constant`). If None, the
        reference default of 1 / (number of cells) = 1/(2*n_bins) is applied
        (this is what their code uses and what reproduces their numbers; the
        paper prose says 1/total_trials, but the code -- which we match --
        uses 1/(2n)).

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
# The unequal-variance SDT likelihood + fit. Same likelihood as the authors'
# reference (uvsdt.R: uvsdt_logL + fit_uvsdt_mle), re-expressed in a smooth
# unconstrained parameterization; per-subject estimates match theirs to
# optimizer precision.
# ---------------------------------------------------------------------------

def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus, log(1 + exp(x)) > 0 for all finite x."""
    return np.logaddexp(0.0, x)


def _softplus_inv(y: np.ndarray) -> np.ndarray:
    """Inverse softplus for y > 0: log(exp(y) - 1), stable via expm1."""
    return np.log(np.expm1(y))


def _unpack(params: np.ndarray, n_ratings: int):
    """Map the unconstrained optimizer vector to (mu, sigma, cri).

    The model is fit in a SMOOTH, UNCONSTRAINED parameterization so the
    objective has no inf/NaN cliff and SciPy's convergence flag is reliable:

        params = [mu, log_sigma, c0, d_1, ..., d_{2n-2}]
        sigma  = exp(log_sigma)                          (> 0 structurally)
        cri[0] = c0
        cri[k] = cri[k-1] - softplus(d_k)                (strictly decreasing)

    Because the criteria are strictly decreasing, Phi(-cri) and
    Phi((mu-cri)/sigma) are strictly increasing, so every predicted cell
    probability (their successive differences) is > 0 by construction --
    no ordering penalty or +inf fallback is needed. This is mathematically
    identical to the free-criterion model; only the path is reparameterized.
    """
    mu = params[0]
    sigma = np.exp(params[1])
    c0 = params[2]
    deltas = params[3:2 * n_ratings + 1]            # length 2n-2
    gaps = _softplus(deltas)                          # strictly positive
    cri = c0 - np.concatenate(([0.0], np.cumsum(gaps)))
    return mu, sigma, cri


def _neg_log_likelihood(params: np.ndarray,
                        nr_s1: np.ndarray,
                        nr_s2: np.ndarray,
                        n_ratings: int) -> float:
    """Negative log-likelihood of the unequal-variance SDT model.

    Same likelihood as the reference uvsdt_logL (uvsdt.R) -- multinomial
    `sum(n * log(p))` over the cells -- but evaluated through the smooth
    `_unpack` parameterization so it is finite and differentiable
    everywhere. Target-absent (S1) ~ N(0, 1); target-present (S2) ~
    N(mu, sigma); cell probabilities are the successive differences of
        pred_far = [0, Phi(0 - cri),          1]
        pred_hr  = [0, Phi((mu - cri)/sigma), 1].
    """
    mu, sigma, cri = _unpack(params, n_ratings)

    pred_far = np.concatenate(([0.0], norm.cdf(0.0 - cri), [1.0]))
    pred_hr = np.concatenate(([0.0], norm.cdf((mu - cri) / sigma), [1.0]))

    # Successive differences are positive by construction; clip only to
    # avoid log(0) from float underflow far out in parameter space.
    p1 = np.maximum(np.diff(pred_far), 1e-300)
    p2 = np.maximum(np.diff(pred_hr), 1e-300)

    ll = np.sum(nr_s1 * np.log(p1)) + np.sum(nr_s2 * np.log(p2))
    if not np.isfinite(ll):
        return 1e12
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

    # Initial guess. Same data-driven seed as uvsdt.R --
    #   rating_far = cumsum(nr_s1)/sum;  rating_hr = cumsum(nr_s2)/sum
    #   mu  = qnorm(rating_hr[n]) - qnorm(rating_far[n]);  sigma = 1.5
    #   cri = -qnorm(rating_far)[1:(2n-1)]   (strictly decreasing)
    # -- re-expressed in the smooth parameterization of `_unpack`:
    #   [mu, log(sigma), c0=cri[0], softplus_inv(gaps between criteria)].
    rating_far = np.cumsum(nr_s1) / nr_s1.sum()
    rating_hr = np.cumsum(nr_s2) / nr_s2.sum()
    # clip only to keep qnorm finite at the seed (does not affect the fit)
    rf = np.clip(rating_far, 1e-6, 1 - 1e-6)
    rh = np.clip(rating_hr, 1e-6, 1 - 1e-6)
    mu0 = norm.ppf(rh[n_ratings - 1]) - norm.ppf(rf[n_ratings - 1])
    cri0 = (-norm.ppf(rf))[:2 * n_ratings - 1]        # decreasing seed
    gaps0 = np.clip(-np.diff(cri0), 1e-6, None)       # positive gaps
    guess = np.concatenate(([mu0, np.log(1.5), cri0[0]],
                            _softplus_inv(gaps0)))

    # Fit the (now smooth, unconstrained) objective with L-BFGS-B. It finds
    # the same optimum as the reference's plain BFGS but, unlike SciPy's BFGS,
    # does not raise spurious "precision loss" (status 2) failures at flat
    # optima -- so res.success is a trustworthy convergence signal here.
    res = minimize(
        _neg_log_likelihood, guess,
        args=(nr_s1, nr_s2, n_ratings),
        method="L-BFGS-B",
        options={"maxiter": 10000},
    )

    mu, sigma, cri = _unpack(res.x, n_ratings)
    mu = float(mu)
    sigma = float(sigma)
    da = mu / np.sqrt((1.0 + sigma ** 2) / 2.0)

    # Output criteria in the reference's order (ascending along the evidence
    # axis): the internal `cri` is strictly decreasing, so reverse it.
    criteria = cri[::-1].copy()

    # Validity: now that the objective is smooth, gate on genuine convergence
    # plus sanity bounds. sigma is structurally > 0 and the criteria are
    # structurally monotone; we still assert finiteness and a sane sigma range
    # so degenerate/non-converged fits (the ones the reference pipeline drops
    # via na.omit / d'>=0) come back valid=False and can be excluded.
    valid = bool(res.success
                 and np.isfinite(mu) and np.isfinite(da)
                 and 0.1 < sigma < 8.0
                 and np.all(np.isfinite(criteria))
                 and np.all(np.diff(criteria) > 0))

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
    confidence No". `add_constant=True` adds 1/(number of cells) = 1/(2n) to
    each cell for stability (their default).

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
    stabilize : float, optional cell constant (default 1/(2*n_bins))
    """
    rating = rt_to_bins(rt, response, n_bins=n_bins)
    return fit_ratings(stimulus, response, rating, n_bins=n_bins,
                       stabilize=stabilize)


