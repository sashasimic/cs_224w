# CS224W milestone draft

[Read the two-page PDF](milestone.pdf), edit [the LaTeX source](milestone.tex), or
see [the pipeline comparison and reproduction appendix](../../docs/pipeline_comparison.md).

The draft follows the Approach / Experiments / Future Work structure of the
user's blank CS224N template. `neurips_2019.sty` is copied from that directory
with its course footer changed to CS224W. Author details remain a placeholder;
CS224W-specific submission requirements have not been provided.

Rebuild from the repository root:

```bash
python scripts/experiments/build_milestone_report.py
```

This reads saved JSON results, regenerates `matched_results.tex`, `numbers.tex`
and the France comparison figure, compiles the PDF twice and checks that it has
exactly two pages. It does not train models. Dependencies: the Python model/
inspection environment, `pdflatex` and `pdfinfo`. Preview figures are available
as [PDF](figures/france_ablation.pdf) and [PNG](figures/france_ablation.png).

The report is an exploratory milestone, not a claim of verified historical
availability. Its main finding is that recent auction prices account for the
largest measured France improvement; the controlled experiment has one seed
and one country. Publication vintages and graph-model evaluation remain open.
