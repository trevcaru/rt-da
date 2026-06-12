"""Real-data validation against the authors' own R (uvsdt.R).

Self-contained: each fixture in validation/fixtures/ holds, per subject, the
authors-faithful count vectors and the R-computed mu/sigma/da/logL. These
tests refit with the package and assert (1) per-subject agreement with R to
optimizer precision and (2) that the published dataset-level da correlations
reproduce. No raw data or R interpreter is needed at test time; see
validation/build_fixtures.py for how the fixtures are regenerated.

Reference (Miyoshi et al. 2026, iScience 29, 114998), Figure 9 da correlations:
Mazor_2020 ~0.90, Sherman JOCN_1 0.73, JOCN_2 0.86.
"""
import json
import os

import numpy as np
import pytest

import rt_da

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _load(name):
    with open(os.path.join(FIX, name)) as fh:
        return json.load(fh)


FIXTURES = ["sherman_jocn1.json", "sherman_jocn2.json", "mazor2020.json"]


def _fit(nr_s1, nr_s2):
    return rt_da.fit_uvsdt_mle(nr_s1, nr_s2, add_constant=True)


# ---------------------------------------------------------------------------
# 1. Per-subject agreement with the authors' R, to optimizer precision.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_per_subject_matches_R(fixture):
    data = _load(fixture)
    da_dev, mu_dev, sig_dev, ll_dev = [], [], [], []
    for s in data["subjects"]:
        R = s["R"]
        for mod, k1, k2 in [("conf", "nr_s1_conf", "nr_s2_conf"),
                            ("rt", "nr_s1_rt", "nr_s2_rt")]:
            if R[f"err_{mod}"]:
                continue  # R's optim errored on this subject/modality
            f = _fit(s[k1], s[k2])
            da_dev.append(abs(f.da - R[f"da_{mod}"]))
            mu_dev.append(abs(f.mu - R[f"mu_{mod}"]))
            sig_dev.append(abs(f.sigma - R[f"sigma_{mod}"]))
            ll_dev.append(abs(f.log_likelihood - R[f"logL_{mod}"]))
    # da is the identified quantity -> tightest; mu/sigma trade along the ridge.
    assert max(da_dev) < 3e-3, f"max da|Δ|={max(da_dev):.2e}"
    assert max(mu_dev) < 1e-2, f"max mu|Δ|={max(mu_dev):.2e}"
    assert max(sig_dev) < 1e-2, f"max sigma|Δ|={max(sig_dev):.2e}"
    assert max(ll_dev) < 1e-2, f"max logL|Δ|={max(ll_dev):.2e}"


# ---------------------------------------------------------------------------
# 2. Dataset-level da correlations reproduce (on R's exclusion set:
#    subjects R could fit, with d' > 0).
# ---------------------------------------------------------------------------

def _correlation_on_R_kept_set(data):
    da_conf, da_rt = [], []
    for s in data["subjects"]:
        R = s["R"]
        if R["err_conf"] or R["err_rt"] or s["dp"] <= 0:
            continue
        fc = _fit(s["nr_s1_conf"], s["nr_s2_conf"])
        fr = _fit(s["nr_s1_rt"], s["nr_s2_rt"])
        da_conf.append(fc.da); da_rt.append(fr.da)
    return np.corrcoef(da_conf, da_rt)[0, 1], len(da_conf)


@pytest.mark.parametrize("fixture,lo,hi", [
    ("mazor2020.json", 0.88, 0.95),     # paper ~0.90
    ("sherman_jocn1.json", 0.70, 0.78),  # paper 0.73 (R excludes subject 7)
    ("sherman_jocn2.json", 0.83, 0.92),  # paper 0.86
])
def test_da_correlation_reproduces(fixture, lo, hi):
    r, n = _correlation_on_R_kept_set(_load(fixture))
    assert lo < r < hi, f"{fixture}: da correlation {r:.3f} (n={n}) outside [{lo},{hi}]"


def test_jocn1_subject7_inclusion_note():
    """Document the one divergence from the paper: the package fits Sherman
    JOCN_1 subject 7 (R's optim errors on it). Including it raises the
    correlation from ~0.73 to ~0.80 -- an optimizer-robustness difference,
    not a fitting error.
    """
    data = _load("sherman_jocn1.json")
    da_conf, da_rt = [], []
    for s in data["subjects"]:
        if s["dp"] <= 0:
            continue
        fc = _fit(s["nr_s1_conf"], s["nr_s2_conf"])
        fr = _fit(s["nr_s1_rt"], s["nr_s2_rt"])
        assert fc.valid and fr.valid  # package fits every subject, incl. id 7
        da_conf.append(fc.da); da_rt.append(fr.da)
    r_all = np.corrcoef(da_conf, da_rt)[0, 1]
    assert 0.78 < r_all < 0.82  # ~0.80 with subject 7 included
