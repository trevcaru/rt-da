"""Dev-time builder for the validation fixtures (NOT run by pytest).

Reads the raw datasets and the authors' R reference output (produced by
docs/_audit_R.R) and writes self-contained JSON fixtures into
validation/fixtures/. Each fixture holds, per subject: the authors-faithful
count vectors (conf + rt) and the R-computed mu/sigma/da/logL for both
modalities, plus R's error flag and the d' used for exclusion. The pytest
suite then needs only these JSON files -- no raw data, no R.

Run from the repo root after regenerating docs/_R_*.csv:
    python validation/build_fixtures.py
"""
import json
import os

import numpy as np
import pandas as pd

from adapters import sherman_counts, mazor_counts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")          # raw datasets (not committed)
RREF = os.path.join(HERE, "r_reference")          # authors' R output (committed)
OUT = os.path.join(HERE, "fixtures")


def _r_table(name):
    df = pd.read_csv(os.path.join(RREF, name))
    return df.set_index("id")


def build_sherman(cond, rcsv, label):
    raw = pd.read_csv(os.path.join(DOCS, "data_Sherman_2016_JOCN(1).csv"))
    raw = raw[raw.Condition == cond]
    R = _r_table(rcsv)
    subjects = []
    for i, d in raw.groupby("Subj_idx"):
        counts = sherman_counts({k: d[k].values for k in
                                 ("Stimulus", "Response", "Confidence", "RT_dec")})
        r = R.loc[i]
        subjects.append(dict(id=int(i), **counts, dp=float(r.dp),
            R=dict(err_conf=bool(r.err_conf), err_rt=bool(r.err_rt),
                   mu_conf=_f(r.mu_conf), sigma_conf=_f(r.sigma_conf),
                   da_conf=_f(r.da_conf), logL_conf=_f(r.logL_conf),
                   mu_rt=_f(r.mu_rt), sigma_rt=_f(r.sigma_rt),
                   da_rt=_f(r.da_rt), logL_rt=_f(r.logL_rt))))
    return dict(dataset=label, n_bins=4,
                note="inverted mapping (nr_s1<-Stimulus==1); 4-level ntile bins",
                subjects=subjects)


def build_mazor():
    raw = pd.read_csv(os.path.join(DOCS, "data_Mazor_2020(1).csv"))
    raw = raw[raw.Condition == "Detection"]
    R = _r_table("mazor2020.csv")
    subjects = []
    for i, d in raw.groupby("Subj_idx"):
        counts = mazor_counts({k: d[k].values for k in
                               ("Stimulus", "Response", "Confidence", "RT_dec")})
        r = R.loc[i]
        subjects.append(dict(id=int(i), **counts, dp=float(r.dp),
            R=dict(err_conf=bool(r.err_conf), err_rt=bool(r.err_rt),
                   mu_conf=_f(r.mu_conf), sigma_conf=_f(r.sigma_conf),
                   da_conf=_f(r.da_conf), logL_conf=_f(r.logL_conf),
                   mu_rt=_f(r.mu_rt), sigma_rt=_f(r.sigma_rt),
                   da_rt=_f(r.da_rt), logL_rt=_f(r.logL_rt))))
    return dict(dataset="Mazor_2020_Detection", n_bins=3,
                note="standard mapping; 3-level ntile bins; reconstructed "
                     "from the 2021 detection script structure",
                subjects=subjects)


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fixtures = {
        "sherman_jocn1.json": build_sherman(1, "sherman_jocn1.csv", "Sherman_2016_JOCN_1"),
        "sherman_jocn2.json": build_sherman(2, "sherman_jocn2.csv", "Sherman_2016_JOCN_2"),
        "mazor2020.json": build_mazor(),
    }
    for fname, data in fixtures.items():
        with open(os.path.join(OUT, fname), "w") as fh:
            json.dump(data, fh, indent=1)
        print(f"wrote {fname}: {len(data['subjects'])} subjects")
