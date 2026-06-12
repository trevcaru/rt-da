"""Print the per-subject R-vs-package diff table from the committed fixtures.

Self-contained: uses the count vectors and the authors' R values baked into
validation/fixtures/*.json, so it needs no raw data and no R interpreter. This
is the human-readable form of what test_real_data.py asserts.

    python validation/compare_against_r.py
"""
import json
import os

import numpy as np

import rt_da

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FILES = [("Sherman JOCN_1", "sherman_jocn1.json"),
         ("Sherman JOCN_2", "sherman_jocn2.json"),
         ("Mazor_2020 Detection", "mazor2020.json")]


def _diffs(data):
    acc = {k: [] for k in ("mu", "sigma", "da", "logL")}
    for s in data["subjects"]:
        R = s["R"]
        for mod, k1, k2 in [("conf", "nr_s1_conf", "nr_s2_conf"),
                            ("rt", "nr_s1_rt", "nr_s2_rt")]:
            if R[f"err_{mod}"]:
                continue
            f = rt_da.fit_uvsdt_mle(s[k1], s[k2], add_constant=True)
            acc["mu"].append(abs(f.mu - R[f"mu_{mod}"]))
            acc["sigma"].append(abs(f.sigma - R[f"sigma_{mod}"]))
            acc["da"].append(abs(f.da - R[f"da_{mod}"]))
            acc["logL"].append(abs(f.log_likelihood - R[f"logL_{mod}"]))
    return acc


def _corr(data):
    dc, dr = [], []
    for s in data["subjects"]:
        R = s["R"]
        if R["err_conf"] or R["err_rt"] or s["dp"] <= 0:
            continue
        dc.append(rt_da.fit_uvsdt_mle(s["nr_s1_conf"], s["nr_s2_conf"]).da)
        dr.append(rt_da.fit_uvsdt_mle(s["nr_s1_rt"], s["nr_s2_rt"]).da)
    return np.corrcoef(dc, dr)[0, 1], len(dc)


if __name__ == "__main__":
    print(f"{'dataset':22s} {'param':6s} {'max|d|':>10s} {'mean|d|':>10s} {'n':>4s}")
    print("-" * 56)
    for label, fname in FILES:
        data = json.load(open(os.path.join(FIX, fname)))
        acc = _diffs(data)
        for p in ("mu", "sigma", "da", "logL"):
            d = np.array(acc[p])
            print(f"{label:22s} {p:6s} {d.max():10.2e} {d.mean():10.2e} {len(d):4d}")
        r, n = _corr(data)
        print(f"{'':22s} da-corr (R kept set) = {r:.3f}  (n={n})")
        print("-" * 56)
