# Complete Model Ranking

This file gives the canonical model results.

The final workflow and `PAPER_TRUTH.json` supply the result values.

## Selected Model

The project selected the Optuna LightGBM model.

| Model | Dataset | Net profit | AUC | Brier score | Decision cutoff | Approval rate | Approved default rate | ROI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Optuna LightGBM `n418 nl194` | Validation | $1,096.4M | 0.9827 | 0.0335 | 0.1488 | 77.8% | 1.09% | 4.70% |
| Optuna LightGBM `n418 nl194` | Test | $1,358.3M | 0.9826 | 0.0339 | 0.1488 | 77.7% | 1.11% | 4.67% |

The manual LightGBM model had the best result before the focused Optuna search.

## Manual Model Ranking

The table ranks each manual model by validation net profit.

| Rank | Model | Model family | Validation profit | AUC | Brier score | Decision cutoff | Approval rate | Approved default rate |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | LGB n300 nl127 | LightGBM | $1,078.6M | 0.9814 | 0.0351 | 0.1934 | 78.6% | 1.39% |
| 2 | LGB n300 nl63 | LightGBM | $1,072.3M | 0.9803 | 0.0365 | 0.1786 | 77.9% | 1.35% |
| 3 | Bagging n100 | Bagging | $1,062.3M | 0.9774 | 0.0399 | 0.1736 | 76.9% | 1.31% |
| 4 | XGB n300 d10 | XGBoost | $1,058.1M | 0.9793 | 0.0367 | 0.1439 | 77.1% | 1.29% |
| 5 | LGB n200 nl31 | LightGBM | $1,055.1M | 0.9772 | 0.0399 | 0.1587 | 76.4% | 1.32% |
| 6 | XGB n200 d8 | XGBoost | $1,045.5M | 0.9770 | 0.0393 | 0.1637 | 76.9% | 1.46% |
| 7 | HGB i300 d63 | HGB | $1,044.4M | 0.9776 | 0.0388 | 0.1835 | 77.9% | 1.61% |
| 8 | HGB i200 d31 | HGB | $1,032.2M | 0.9745 | 0.0420 | 0.1667 | 76.5% | 1.57% |
| 9 | AdaBoost n100 | AdaBoost | $1,020.0M | 0.9712 | 0.1561 | 0.4562 | 75.9% | 1.71% |
| 10 | MLP 256x128x64x32 | MLP | $978.5M | 0.9595 | 0.0514 | 0.0992 | 75.3% | 2.24% |
| 11 | MLP 256x128x64 | MLP | $972.3M | 0.9573 | 0.0527 | 0.1667 | 77.3% | 2.68% |
| 12 | MLP 128x64x32 | MLP | $970.9M | 0.9580 | 0.0521 | 0.1240 | 76.1% | 2.42% |
| 13 | RF n100 d16 l25 | Random Forest | $947.2M | 0.9541 | 0.0592 | 0.1667 | 73.2% | 2.23% |
| 14 | RF n100 d16 l50 | Random Forest | $936.0M | 0.9521 | 0.0606 | 0.1538 | 71.5% | 2.10% |
| 15 | RF n200 d16 l50 | Random Forest | $934.8M | 0.9520 | 0.0608 | 0.1538 | 71.5% | 2.10% |
| 16 | Ridge C1 | Logistic | $791.4M | 0.8745 | 0.0899 | 0.2232 | 72.9% | 5.08% |
| 17 | Ridge C10 | Logistic | $791.3M | 0.8745 | 0.0898 | 0.2232 | 73.0% | 5.07% |
| 18 | Ridge C0.1 | Logistic | $790.8M | 0.8745 | 0.0899 | 0.2182 | 72.4% | 4.99% |
| 19 | LDA | LDA | $769.3M | 0.8606 | 0.0959 | 0.2083 | 71.6% | 5.62% |
| 20 | QDA r0.1 | QDA | $697.4M | 0.7944 | 0.1873 | 0.0893 | 57.8% | 6.01% |
| 21 | QDA r0.2 | QDA | $693.0M | 0.7887 | 0.1745 | 0.1042 | 59.0% | 6.23% |

## External Feature Result

The table shows the effect of each feature group.

| Feature group | Feature count | Validation profit | AUC | Brier score | Decision cutoff | Approval rate | Approved default rate | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SBA data | 16 | $1,087.9M | 0.9809 | 0.0355 | 0.1687 | 78.1% | 1.26% | 4.64% |
| SBA and external data | 31 | $1,093.4M | 0.9825 | 0.0337 | 0.1637 | 78.2% | 1.21% | 4.65% |
| SBA, external, and interaction data | 44 | $1,096.4M | 0.9827 | 0.0335 | 0.1488 | 77.8% | 1.09% | 4.70% |

The full feature set increased validation profit by approximately `$8.5M`.

The full feature set decreased the approved default rate from `1.26%` to `1.09%`.

## Robustness Results

| Check | Result |
|---|---|
| Multiple random seeds | Mean profit was `$1,091.7M`, with a standard deviation of `$4.1M` |
| Era-forward test | Pre-to-crisis AUC was `0.960` |
| Era-forward test | Pre-and-crisis-to-post AUC was `0.970` |
| Profit sensitivity | The model had a positive profit for five default-loss assumptions |
| Year removal | Profit decreased by approximately `$1.1M`, and AUC decreased by `0.0006` |
| Calibration | Calibration changed profit by less than `$1M` near the decision cutoff |

## Profit Rule

The validation set selects the model and the decision cutoff.

| Decision | Actual outcome | Profit value |
|---|---|---:|
| Approve | Paid in full | `+5% × DisbursementGross` |
| Approve | Default | `-25% × DisbursementGross` |
| Deny | Paid in full or default | `$0` |

The profit calculation uses `DisbursementGross`. The feature matrix does not contain this field.

## Result Control

Do not change a result value without a source-of-truth check.

Use `PAPER_TRUTH.json` in the LaTeX source package for this check.
