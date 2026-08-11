# SBA Loan Default Prediction

Profit-based machine learning project for SBA loan approval decisions. The project compares classical models, tree ensembles, neural networks, external economic context, and Optuna-tuned LightGBM under a business objective: approve loans only when the expected lending profit is positive.

## Final Result

The final model is an Optuna-tuned LightGBM classifier using approval-time SBA features plus selected external context.

| Split | Net profit | AUC | Brier | Approval rate | Approved default rate | ROI |
|---|---:|---:|---:|---:|---:|---:|
| Validation | $1,096.4M | 0.9827 | 0.0335 | 77.8% | 1.09% | 4.70% |
| Test | $1,358.3M | 0.9826 | 0.0339 | 77.7% | 1.11% | 4.67% |

Final decision cutoff: approve when predicted default probability is below `0.1488`.

## Business Rule

The evaluation optimizes validation profit, not ROC AUC alone.

| Decision | Actual outcome | Profit rule |
|---|---|---:|
| Approve | Paid in full | `+5% * DisbursementGross` |
| Approve | Default | `-25% * DisbursementGross` |
| Deny | Any outcome | `$0` |

`DisbursementGross` is used only to calculate retrospective profit. It is not used as a model predictor.

## Modeling Scope

The workflow evaluates a mix of model families:

| Family | Examples |
|---|---|
| Linear / discriminant | Ridge logistic regression, LDA, QDA |
| Tree ensembles | Random Forest, Bagging, AdaBoost, HistGradientBoosting |
| Boosting | XGBoost, LightGBM |
| Neural network | MLP |
| Tuning | Optuna profit-based LightGBM |

The final model uses 44 pre-approval features: 37 numeric and 7 categorical fields before encoding.

## External Data

External data is merged only when it is observable at or before the loan approval period. The goal is not to leak future outcomes, but to test whether the SBA application becomes stronger when local economic context is added.

| Block | Purpose |
|---|---|
| BLS unemployment | State labor-market stress |
| BLS QCEW | State-industry employment and wage conditions |
| FHFA HPI | Housing-market weakness |
| FEMA disasters | Local shock exposure |
| Census CBP | Local industry thickness |
| Engineered interactions | Housing, disaster, industry, and small-firm risk stacks |

The final comparison found that external features improved the Optuna LightGBM validation profit from `$1,087.9M` to `$1,096.4M`, a lift of about `$8.5M`. The larger value of the external block is interpretability: it helps explain which local stress channels matter around the approval decision.

## Leakage Controls

The modeling code excludes outcome and post-approval fields before training:

| Excluded field | Reason |
|---|---|
| `MIS_Status` | Target label |
| `ChgOffDate` | Observed after charge-off |
| `ChgOffPrinGr` | Observed after charge-off |
| `BalanceGross` | Post-approval balance information |
| `DisbursementGross` | Kept only for profit weighting, not prediction |

## Repository Structure

| Path | Contents |
|---|---|
| `Main Workflow.py` | Main modeling workflow used for the project |
| `notebooks/final_workflow.ipynb` | Final notebook version for review |
| `reports/` | Final paper, presentation, and LaTeX source package |
| `TECHNICAL_DOCUMENT.md` | ASD-STE100-style technical description and operating instructions |
| `COMPLETE_MODEL_RANKING.md` | Canonical model leaderboard and final metric summary |
| `CONTRIBUTION_STATEMENT.md` | Team contribution percentages |

Large local files are intentionally excluded from GitHub: raw SBA data, external-data caches, parquet merges, temporary research outputs, and LaTeX build artifacts.

## Reproducibility Notes

The raw SBA dataset is not committed because of size. To rerun locally, place `SBAnational.csv` in the project root and use the final notebook or main workflow. External merged artifacts can be regenerated from the workflow scripts, but the portfolio repository keeps only final deliverables and source code.

## Final Deliverables

- [Technical description and operating instructions](TECHNICAL_DOCUMENT.md)
- [Research paper](reports/Group%202%20ML%20Final%20Project%20Research%20Paper.pdf)
- [Presentation deck](reports/Group%202%20ML%20Final%20Project%20Presentation.pdf)
- [LaTeX source package](reports/Group2_LaTeX_Source_Package_20260722.zip)

## Ethics and Use

This is a course project and portfolio artifact, not a production lending system. A real deployment would require compliance review, fairness testing, adverse-action explanation, monitoring for drift, and manual review for borderline decisions.
