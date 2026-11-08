# Milestone v3

[Two-page PDF](milestone.pdf) · [LaTeX source](milestone.tex) ·
[Exact table values](results.json) · [Source manifest](sources.json)

Seven methods: weekly persistence; per-zone and pooled richer XGBoost; static
pooled MLP; the verified partner-inspired residual GRU+MLP; self-loop and real GAT.
The static learned controls use five seeds and identical richer inputs. The GRU
adapts Sasha's recipe on our canonical data, using one seed, multivariate histories,
a persistence-residual target and residual-contrast loss. It is not a controlled
single-change architecture comparison, or the partner's original 43-zone score.

Table 1 uses the same 49-market cohort and common persistence hours throughout.
All-target RMSE is separately labelled. Smallest means are bold; seed SD is not
statistical significance. The affected original fused-backend GRUs are excluded.
ERA5 and historical source vintages remain conditional, and the test period was
previously explored. Author details remain placeholders.

Rebuild from the repository root:

```bash
.venv/bin/python scripts/experiments/build_milestone_v3.py
```

The builder checks verified result identities, reconciles five-seed table scores
against zone summaries, and checks selected job target signatures against the
accepted GRU cohort. It compiles in a temporary directory, requires exactly two
pages without overflowing boxes, and publishes the PDF atomically. No training
or GPU use. Dependencies: the project environment, `pdflatex`, `pdfinfo` and
`pdftotext`. Earlier milestone PDF hashes must remain unchanged.

[All experiments and audits](../../docs/experiments.md) ·
[Frozen v1](../milestone_v1/README.md) · [Historical v2](../milestone_v2/README.md)
