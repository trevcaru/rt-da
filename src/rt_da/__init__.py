"""
rt_da - RT-based unequal-variance signal detection (da)
=======================================================

A clean, validated Python implementation of the method in:

    Miyoshi, K., Rahnev, D., & Lau, H. (2026). Correcting for unequal variance
    in signal detection models using response time. iScience 29, 114998.
    https://doi.org/10.1016/j.isci.2026.114998

In a yes/no detection task the target-present internal distribution is usually
more variable than the target-absent one (sigma > 1). Conventional d' assumes
equal variance and misestimates sensitivity. The unequal-variance index da
corrects this, and this package estimates it from response times (RT) -- using
faster RT as a proxy for higher confidence -- as well as from confidence
ratings, fitting the same model to either.

Quick start
-----------
    import rt_da

    # from trial-level data
    fit = rt_da.rt_da(stimulus, response, rt, n_bins=3)
    print(fit.da, fit.sigma, fit.mu)

    # compare RT-based vs confidence-based estimates
    dual = rt_da.compare_rt_confidence(stimulus, response, rt, confidence)
    print(dual.summary())

    # batch over subjects in a tidy DataFrame
    table = rt_da.fit_group(df, subject="subj", stimulus="stim",
                            response="resp", rt="rt", confidence="conf")

Validation
----------
`fit_uvsdt_mle` reproduces the reference implementation's published example
exactly (mu=1.2314, sigma=1.2523, da=1.0867, logL=-315.88) and matches its
per-subject estimates on real datasets (see tests/).
"""

from .core import (
    SDTFit,
    rt_to_bins,
    build_roc_table,
    fit_uv_sdt,
    fit_uvsdt_mle,
    fit_ratings,
    rt_da,
)
from .diagnostics import (
    RTValidity,
    rt_validity,
    rt_da_ci,
    DualEstimate,
    compare_rt_confidence,
)
from .group import fit_group
from .simulate import simulate_detection

__version__ = "0.1.0"

__all__ = [
    "SDTFit",
    "rt_to_bins",
    "build_roc_table",
    "fit_uv_sdt",
    "fit_uvsdt_mle",
    "fit_ratings",
    "rt_da",
    "RTValidity",
    "rt_validity",
    "rt_da_ci",
    "DualEstimate",
    "compare_rt_confidence",
    "fit_group",
    "simulate_detection",
    "__version__",
]
