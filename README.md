# rt-da

**RT-based unequal-variance signal detection (da) in Python.**

In a yes/no detection task, the target-present internal distribution is
usually *more variable* than the target-absent one (SD ratio σ > 1).
Conventional `d′` assumes equal variance and so **systematically misestimates
sensitivity** in a criterion-dependent way. The unequal-variance index `da`
fixes this:

```
da = μ / sqrt((1 + σ²) / 2)
```

Fitting `da` normally needs multiple points in ROC space — usually from
confidence ratings. This package also estimates it from **response times**,
using *faster RT as a proxy for higher confidence*, following:

> Miyoshi, K., Rahnev, D., & Lau, H. (2026). *Correcting for unequal variance
> in signal detection models using response time.* iScience 29, 114998.
> https://doi.org/10.1016/j.isci.2026.114998

This is an independent Python implementation. The authors' original R code is
at <https://github.com/kiyomiyoshi/rt_type1_roc>.

---

## Why this package

- **Python-native.** The reference implementation is R; this fits cleanly into
  PsychoPy / pandas / SciPy workflows.
- **Validated against the published method.** `fit_uvsdt_mle` reproduces the
  reference example *exactly* and matches its per-subject estimates on real
  datasets (Mazor 2020, Sherman 2016). See `tests/`.
- **Does more than the paper packages:** RT-validity diagnostics, bootstrap
  confidence intervals, tidy group-level batch fitting, and publication-style
  plots — all included, all free.
- **Honest about RT.** Every dual fit reports the RT-based *and*
  confidence-based estimate side by side, plus whether RT actually tracks
  confidence in *your* data (it doesn't always — e.g. memory tasks).

## Install

```bash
pip install rt-da            # core (numpy, scipy, pandas)
pip install "rt-da[plots]"   # + matplotlib for the plotting helpers
```

## Quick start

```python
import rt_da

# Trial-level arrays: stimulus (1=present, 0=absent),
# response (1=yes, 0=no), rt (seconds), confidence (1..n)
fit = rt_da.rt_da(stimulus, response, rt, n_bins=3)
print(fit.da, fit.sigma, fit.mu, fit.dprime)

# Compare RT-based vs confidence-based estimates on the same trials
dual = rt_da.compare_rt_confidence(stimulus, response, rt, confidence)
print(dual.summary())

# Is RT a valid proxy in your data?
print(rt_da.rt_validity(rt, confidence))

# Confidence intervals
ci = rt_da.rt_da_ci(stimulus, response, rt, n_boot=2000)
print(ci["da"])   # (estimate, lo, hi)

# Batch over subjects in a tidy DataFrame
table = rt_da.fit_group(df, subject="subj", stimulus="stim",
                        response="resp", rt="rt", confidence="conf")
```

### Working directly with count vectors

If you already have the response-frequency vectors (S1 = target-absent,
S2 = target-present), ordered from highest support for "yes" to lowest:

```python
fit = rt_da.fit_uvsdt_mle(nr_s1, nr_s2, add_constant=True)
# -> SDTFit(mu, sigma, da, criteria, log_likelihood, ...)
```

## Validation

`fit_uvsdt_mle([10,7,16,27,29,10], [43,21,10,12,8,3])` returns
mu = 1.2314, sigma = 1.2523, da = 1.0867, logL = −315.88 — matching the
reference implementation's published example to the reported digits. Run the
suite:

```bash
pip install "rt-da[dev]"
pytest
```

## Notes & caveats

- RT-based `da` is a *pragmatic alternative* when confidence isn't available,
  not a universal replacement. The original paper found it works well for
  perceptual detection but **underestimates** performance on memory tasks,
  where RT carries less information about accuracy. Use `rt_validity()`.
- `add_constant=True` adds `1/(2n)` to each cell for stability, matching the
  reference implementation.

## Citing

If you use this package, please cite the original method (Miyoshi, Rahnev &
Lau, 2026, above).

## Support

This package is free and open-source, and will stay that way. If it saved you
time or helped your work, you can support maintenance:

- **Ko-fi (one-time tip):** https://ko-fi.com/trevcaru

No pressure — the package is free regardless.

## License

MIT — see [LICENSE](LICENSE).
