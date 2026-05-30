"""Tidy group-level batch fitting over a DataFrame of subjects."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .core import rt_da, fit_ratings


def fit_group(df: pd.DataFrame,
              subject: str,
              stimulus: str,
              response: str,
              rt: Optional[str] = None,
              confidence: Optional[str] = None,
              n_bins: int = 3,
              condition: Optional[str] = None,
              stabilize: Optional[float] = None) -> pd.DataFrame:
    """Fit da for every subject (and condition) in a tidy DataFrame.

    Supply `rt=` for RT-based da, `confidence=` for confidence-based da, or
    BOTH to get both in the same output (the paper's RT-vs-confidence
    comparison becomes one call).

    Returns one row per subject x condition with mu/sigma/da/dprime/criterion
    for each supplied modality, suffixed _rt and/or _conf.
    """
    if rt is None and confidence is None:
        raise ValueError("provide rt=, confidence=, or both")

    group_cols = [subject] + ([condition] if condition else [])
    rows = []
    for keys, g in df.groupby(group_cols, dropna=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        stim = g[stimulus].to_numpy()
        resp = g[response].to_numpy()

        if rt is not None:
            try:
                f = rt_da(stim, resp, g[rt].to_numpy(),
                          n_bins=n_bins, stabilize=stabilize)
                row.update({f"{k}_rt": v for k, v in f.as_dict().items()
                            if k in ("mu", "sigma", "da", "dprime",
                                     "criterion", "converged")})
            except Exception:
                row.update({f"{k}_rt": np.nan for k in
                            ("mu", "sigma", "da", "dprime", "criterion")})
                row["converged_rt"] = False

        if confidence is not None:
            try:
                f = fit_ratings(stim, resp, g[confidence].to_numpy(),
                               n_bins=n_bins, stabilize=stabilize)
                row.update({f"{k}_conf": v for k, v in f.as_dict().items()
                            if k in ("mu", "sigma", "da", "dprime",
                                     "criterion", "converged")})
            except Exception:
                row.update({f"{k}_conf": np.nan for k in
                            ("mu", "sigma", "da", "dprime", "criterion")})
                row["converged_conf"] = False

        rows.append(row)
    return pd.DataFrame(rows)

