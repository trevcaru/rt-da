"""Simulator for detection experiments with known (mu, sigma).

Used for tests, parameter-recovery demos, and documentation examples.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm  # noqa: F401  (kept for parity / future use)


def simulate_detection(n_trials: int = 400,
                       mu: float = 1.5,
                       sigma: float = 1.5,
                       criterion: float = 0.0,
                       rt_noise: float = 0.15,
                       p_present: float = 0.5,
                       seed: Optional[int] = None) -> pd.DataFrame:
    """Simulate a yes/no detection experiment with a known (mu, sigma).

    The latent evidence x is drawn from N(0,1) on absent trials and
    N(mu, sigma) on present trials. Response = "yes" if x > criterion.
    RT is generated so that stronger evidence (further from the criterion)
    is FASTER -- giving RT genuine information about the choice, exactly the
    assumption the method relies on. Confidence is a 3-level discretization
    of |x - criterion| for cross-checking RT vs confidence.

    Returns a tidy DataFrame: stimulus, response, rt, confidence, evidence.
    """
    rng = np.random.default_rng(seed)
    present = rng.random(n_trials) < p_present
    x = np.where(present,
                 rng.normal(mu, sigma, n_trials),
                 rng.normal(0.0, 1.0, n_trials))
    response = (x > criterion).astype(int)

    # Distance from criterion = evidence strength for the chosen response.
    strength = np.abs(x - criterion)
    # RT decreases monotonically with evidence strength (a drift-diffusion-like
    # relationship: stronger evidence -> faster decision), with multiplicative
    # lognormal noise. Lower rt_noise => RT is a cleaner proxy for the latent
    # strength and RT-based recovery tightens toward the confidence-based fit.
    base_rt = 0.3 + 0.8 * np.exp(-0.9 * strength)
    rt = base_rt * np.exp(rng.normal(0.0, rt_noise, n_trials))

    # 3-level confidence from tertiles of strength (per the paper's binning).
    try:
        conf = pd.qcut(strength, 3, labels=[1, 2, 3]).astype(int)
    except ValueError:
        conf = np.full(n_trials, 2, dtype=int)

    return pd.DataFrame({
        "stimulus": present.astype(int),
        "response": response,
        "rt": rt,
        "confidence": np.asarray(conf, dtype=int),
        "evidence": x,
    })

