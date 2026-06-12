"""Authors-faithful preprocessing adapters for the real-data validation.

These reproduce the count-vector construction in the reference R scripts
(kiyomiyoshi/rt_type1_roc: Sherman_2016_JOCN_1.R / _2.R and the Mazor_2020
detection analysis) line for line, so the package can be checked against the
authors' own per-subject estimates. The RT binning replicates dplyr::ntile
exactly (equal-sized groups, larger groups first, ties broken by row order) --
NOT the package's internal pd.qcut binning, because that is what the R scripts
use.
"""
from __future__ import annotations

import numpy as np


def ntile(x, n: int) -> np.ndarray:
    """Faithful base port of dplyr::ntile.

    Returns integer labels 1..n (0 for NaN). Larger groups come first; ties
    are broken by order of appearance (stable sort on value).
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(np.where(np.isnan(x), np.inf, x), kind="stable")
    len_ = int(np.isfinite(x[order]).sum())
    out = np.zeros(len(x), dtype=int)
    if len_ == 0:
        return out
    n_larger = len_ % n
    larger_size = int(np.ceil(len_ / n))
    smaller_size = int(np.floor(len_ / n))
    larger_threshold = larger_size * n_larger
    ranks = np.arange(1, len_ + 1)
    bins = np.where(
        ranks <= larger_threshold,
        (ranks - 1) // max(larger_size, 1) + 1,
        (ranks - larger_threshold - 1) // max(smaller_size, 1) + n_larger + 1,
    )
    out[order[:len_]] = bins
    return out


def sherman_counts(d):
    """Build (nr_s1, nr_s2) count vectors for one Sherman subject.

    `d` is a dict/DataFrame-like with arrays Stimulus, Response, Confidence,
    RT_dec. Uses the inverted mapping (nr_s1 <- Stimulus==1, nr_s2 <-
    Stimulus==0) and 4-level binning, exactly as Sherman_2016_JOCN_*.R.

    Returns dict with conf/rt count vectors (each length 8).
    """
    S = np.asarray(d["Stimulus"]); R = np.asarray(d["Response"])
    C = np.asarray(d["Confidence"]); RT = np.asarray(d["RT_dec"], float)

    def cc(stim, resp, conf):
        return int(np.sum((S == stim) & (R == resp) & (C == conf)))
    nr_s1_conf = [cc(1,0,4),cc(1,0,3),cc(1,0,2),cc(1,0,1),
                  cc(1,1,1),cc(1,1,2),cc(1,1,3),cc(1,1,4)]
    nr_s2_conf = [cc(0,0,4),cc(0,0,3),cc(0,0,2),cc(0,0,1),
                  cc(0,1,1),cc(0,1,2),cc(0,1,3),cc(0,1,4)]

    rb = ntile(RT, 4)
    def cr(stim, resp, bn):
        return int(np.sum((S == stim) & (R == resp) & (rb == bn)))
    nr_s1_rt = [cr(1,0,1),cr(1,0,2),cr(1,0,3),cr(1,0,4),
                cr(1,1,4),cr(1,1,3),cr(1,1,2),cr(1,1,1)]
    nr_s2_rt = [cr(0,0,1),cr(0,0,2),cr(0,0,3),cr(0,0,4),
                cr(0,1,4),cr(0,1,3),cr(0,1,2),cr(0,1,1)]
    return dict(nr_s1_conf=nr_s1_conf, nr_s2_conf=nr_s2_conf,
                nr_s1_rt=nr_s1_rt, nr_s2_rt=nr_s2_rt)


def mazor_counts(d):
    """Build (nr_s1, nr_s2) count vectors for one Mazor_2020 detection subject.

    Standard (non-inverted) mapping: nr_s2 <- Stimulus==1 (present),
    nr_s1 <- Stimulus==0 (absent); Response==1 == "yes". 3-level ntile bins,
    matching the structure of the authors' Mazor detection analysis.

    Returns dict with conf/rt count vectors (each length 6).
    """
    S = np.asarray(d["Stimulus"]); R = np.asarray(d["Response"])
    C = np.asarray(d["Confidence"], float); RT = np.asarray(d["RT_dec"], float)
    Cb = ntile(C, 3); Rb = ntile(RT, 3)

    def cc(stim, resp, bn):
        return int(np.sum((S == stim) & (R == resp) & (Cb == bn)))
    def cr(stim, resp, bn):
        return int(np.sum((S == stim) & (R == resp) & (Rb == bn)))
    nr_s1_conf = [cc(0,1,3),cc(0,1,2),cc(0,1,1),cc(0,0,1),cc(0,0,2),cc(0,0,3)]
    nr_s2_conf = [cc(1,1,3),cc(1,1,2),cc(1,1,1),cc(1,0,1),cc(1,0,2),cc(1,0,3)]
    nr_s1_rt = [cr(0,1,1),cr(0,1,2),cr(0,1,3),cr(0,0,3),cr(0,0,2),cr(0,0,1)]
    nr_s2_rt = [cr(1,1,1),cr(1,1,2),cr(1,1,3),cr(1,0,3),cr(1,0,2),cr(1,0,1)]
    return dict(nr_s1_conf=nr_s1_conf, nr_s2_conf=nr_s2_conf,
                nr_s1_rt=nr_s1_rt, nr_s2_rt=nr_s2_rt)
