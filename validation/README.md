# Real-data validation

This suite checks `rt-da` against the authors' own R implementation
(`uvsdt.R`, kiyomiyoshi/rt_type1_roc) on the Mazor (2020) and Sherman (2016)
datasets, at the single-subject level.

## What runs under pytest

- **`test_real_data.py`** — for every subject it refits the authors-faithful
  count vectors with `fit_uvsdt_mle` and asserts agreement with the R-computed
  μ, σ, da, and log-likelihood (da within 2.5e-3 across 80 subject-fits), then
  checks that the dataset-level RT-vs-confidence da correlations reproduce
  (Mazor ≈0.91, JOCN_1 0.73 on the authors' exclusion set, JOCN_2 0.87).

The tests read only `fixtures/*.json`, which bake in the derived count vectors
**and** the R reference values — so no raw data and no R interpreter are
needed to run them.

## Files

| path | role |
|------|------|
| `fixtures/*.json` | self-contained test fixtures (count vectors + R values) |
| `adapters.py` | authors-faithful preprocessing (ntile + count construction) |
| `r_reference/*.csv` | per-subject output of the authors' `uvsdt.R` (ground truth) |
| `r_reference/generate_r_reference.R` | regenerates those CSVs from the raw data |
| `build_fixtures.py` | rebuilds `fixtures/` from raw data + `r_reference/` |
| `compare_against_r.py` | prints the per-subject R-vs-package diff table |

## Regenerating (dev only)

Needs the raw dataset CSVs in `../docs/` (not committed) and R with `uvsdt.R`:

```bash
cd validation/r_reference && Rscript generate_r_reference.R && cd ..
python build_fixtures.py
```

## The one divergence from the paper

Sherman JOCN_1 subject 7: R's optimizer **errors** on this subject, so the
reference pipeline drops it (n=17, correlation 0.73). This package fits it
without error; including it raises the correlation to ~0.80. This is an
optimizer-robustness difference, not a fitting error — the per-subject
estimates match R wherever R produces one. Both values are covered by the
tests.
