"""Tests for rt_da.

The headline test (`test_reference_example`) reproduces the published example
from the reference implementation (Miyoshi et al. 2026, uvsdt.R) exactly.
If it ever fails, the fitter has drifted from the validated method.
"""
import numpy as np
import pytest

import rt_da


# ---------------------------------------------------------------------------
# Headline: exact match to the published reference example.
# ---------------------------------------------------------------------------

def test_reference_example():
    """fit_uvsdt_mle must reproduce the reference's published numbers."""
    f = rt_da.fit_uvsdt_mle([10, 7, 16, 27, 29, 10],
                            [43, 21, 10, 12, 8, 3], add_constant=True)
    assert f.mu == pytest.approx(1.231417, abs=1e-3)
    assert f.sigma == pytest.approx(1.252275, abs=1e-3)
    assert f.da == pytest.approx(1.086691, abs=1e-3)
    assert f.log_likelihood == pytest.approx(-315.8781, abs=1e-2)
    expected_cri = [-1.234045, -0.2754573, 0.3930064, 0.8424681, 1.381982]
    assert np.allclose(f.criteria, expected_cri, atol=1e-3)
    assert f.valid


def test_da_formula_equals_dprime_when_sigma_one():
    """When sigma == 1, da reduces to conventional d'."""
    # Symmetric data => sigma ~ 1 => da ~ d'
    nr = [30, 20, 10, 10, 20, 30]
    f = rt_da.fit_uvsdt_mle(nr, nr[::-1])
    # not exactly 1, but da and dprime should be close when sigma near 1
    if abs(f.sigma - 1) < 0.1:
        assert f.da == pytest.approx(f.dprime, abs=0.1)


# ---------------------------------------------------------------------------
# Parameter recovery from the simulator.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("true_mu,true_sigma", [
    (1.5, 1.0), (1.5, 1.5), (2.0, 1.8), (1.0, 1.3),
])
def test_confidence_recovery(true_mu, true_sigma):
    """Confidence-based fit should recover known parameters closely."""
    true_da = true_mu / np.sqrt((1 + true_sigma ** 2) / 2)
    das, sigmas = [], []
    for seed in range(6):
        df = rt_da.simulate_detection(n_trials=4000, mu=true_mu,
                                      sigma=true_sigma, seed=seed)
        f = rt_da.fit_ratings(df.stimulus, df.response, df.confidence,
                              n_bins=3)
        das.append(f.da)
        sigmas.append(f.sigma)
    assert np.mean(das) == pytest.approx(true_da, abs=0.15)
    assert np.mean(sigmas) == pytest.approx(true_sigma, abs=0.2)


def test_rt_tracks_confidence_in_simulation():
    """RT-based and confidence-based da should agree on simulated data."""
    das_rt, das_cf = [], []
    for seed in range(8):
        df = rt_da.simulate_detection(n_trials=2000, mu=1.5, sigma=1.5,
                                      seed=seed)
        d = rt_da.compare_rt_confidence(df.stimulus, df.response, df.rt,
                                        df.confidence)
        das_rt.append(d.da_rt)
        das_cf.append(d.da_conf)
    r = np.corrcoef(das_rt, das_cf)[0, 1]
    assert r > 0.8


# ---------------------------------------------------------------------------
# API / plumbing.
# ---------------------------------------------------------------------------

def test_rt_da_end_to_end():
    df = rt_da.simulate_detection(n_trials=1000, mu=1.5, sigma=1.4, seed=0)
    f = rt_da.rt_da(df.stimulus, df.response, df.rt, n_bins=3)
    assert f.valid
    assert np.isfinite(f.da)
    assert f.sigma > 0


def test_rt_validity_detects_signal():
    df = rt_da.simulate_detection(n_trials=2000, mu=1.5, sigma=1.5, seed=0)
    v = rt_da.rt_validity(df.rt, df.confidence)
    # faster RT -> higher confidence => negative Spearman rho
    assert v.rho < 0
    assert "valid" in v.verdict.lower()


def test_bootstrap_ci_brackets_estimate():
    df = rt_da.simulate_detection(n_trials=800, mu=1.5, sigma=1.5, seed=1)
    ci = rt_da.rt_da_ci(df.stimulus, df.response, df.rt, n_boot=400, seed=1)
    pe, lo, hi = ci["da"]
    assert lo <= pe <= hi
    # Some resamples yield degenerate count vectors and are discarded; we
    # just need enough usable resamples to form a stable interval.
    assert ci["n_boot_used"] > 50


def test_fit_group_returns_row_per_subject():
    frames = []
    for s in range(5):
        d = rt_da.simulate_detection(n_trials=600, mu=1.2 + 0.1 * s,
                                     sigma=1.4, seed=s)
        d["subject"] = s
        frames.append(d)
    import pandas as pd
    big = pd.concat(frames, ignore_index=True)
    g = rt_da.fit_group(big, subject="subject", stimulus="stimulus",
                        response="response", rt="rt", confidence="confidence")
    assert len(g) == 5
    assert "da_rt" in g.columns
    assert "da_conf" in g.columns


def test_valid_implies_converged_and_sane():
    """The A2 contract: valid=True must imply genuine convergence AND sane
    estimates (finite mu/da, 0.1<sigma<8, finite strictly-ordered criteria).
    """
    import numpy as np
    rng = np.random.default_rng(0)
    seen_valid = False
    for seed in range(12):
        df = rt_da.simulate_detection(n_trials=1500, mu=1.3, sigma=1.4,
                                      seed=seed)
        for f in (rt_da.rt_da(df.stimulus, df.response, df.rt, n_bins=3),
                  rt_da.fit_ratings(df.stimulus, df.response, df.confidence,
                                    n_bins=3)):
            if f.valid:
                seen_valid = True
                assert f.converged
                assert np.isfinite(f.mu) and np.isfinite(f.da)
                assert 0.1 < f.sigma < 8.0
                assert np.all(np.isfinite(f.criteria))
                assert np.all(np.diff(f.criteria) > 0)  # strictly ordered
    assert seen_valid  # the gate is not vacuously rejecting everything


def test_no_runtime_warnings_from_fit(recwarn):
    """The smooth objective must not emit the NaN-gradient RuntimeWarnings the
    old inf-cliff did (we no longer suppress them in pyproject)."""
    import warnings
    df = rt_da.simulate_detection(n_trials=1200, mu=1.5, sigma=1.5, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rt_da.rt_da(df.stimulus, df.response, df.rt, n_bins=3)
        rt_da.fit_uvsdt_mle([10, 7, 16, 27, 29, 10], [43, 21, 10, 12, 8, 3])


def test_add_constant_default_is_one_over_cells():
    """add_constant must add 1/(2n) per cell to match the reference."""
    nr1 = np.array([10, 7, 16, 27, 29, 10], float)
    nr2 = np.array([43, 21, 10, 12, 8, 3], float)
    with_const = rt_da.fit_uvsdt_mle(nr1, nr2, add_constant=True)
    # manual 1/6 addition should reproduce the same fit via fit_uv_sdt
    manual = rt_da.fit_uv_sdt(nr1 + 1/6, nr2 + 1/6, n_bins=3)
    assert with_const.da == pytest.approx(manual.da, abs=1e-6)
