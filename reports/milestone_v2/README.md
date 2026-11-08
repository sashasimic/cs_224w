# Milestone v2

**Historical snapshot:** the negative GRU interpretation in this PDF is
superseded by the November 9 training-shaped backend audit. The scores remain
archived; corrected original-family training is pending. Read the
[current experiment record](../../docs/experiments.md) for all programmes and
the accepted residual-GRU result. The PDF is preserved unchanged.

[Two-page PDF](milestone.pdf) · [LaTeX](milestone.tex) · [Frozen v1](../milestone_v1/milestone.pdf) · [Detailed comparison](../../docs/controlled_study_comparison.md)

Rebuild with `.venv/bin/python scripts/experiments/build_milestone_v2.py`. All 100 jobs must pass artifact verification first. No models are fitted. Author details remain placeholders.

Presentation revision: Table 1 groups methods by class, includes weekly persistence and relative MAE, and bolds the smallest mean in each column. Future work now proposes a daily prospective forecasting service. Numerical model results are unchanged. The [availability audit](../../docs/leakage_audit.md) still applies; this is a conditional retrospective report. See [metrics and prospective deployment](../../docs/metrics_and_live_forecasting.md).
