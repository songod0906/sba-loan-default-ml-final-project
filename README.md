# SBA Loan Default Prediction

This project estimates the default probability for an SBA loan.

The project uses validation net profit to select the model and the decision cutoff.

## Important Use Limit

**IMPORTANT:** Do not use this model for a real loan decision.

This model is a course project and a portfolio artifact. It is not a production lending system.

A production system must include legal review, fairness tests, decision explanations, drift monitoring, and manual review.

## Final Model

The final model is an Optuna-tuned LightGBM classifier. It uses 44 features that were available before loan approval.

| Dataset | Net profit | AUC | Brier score | Approval rate | Approved default rate | ROI |
|---|---:|---:|---:|---:|---:|---:|
| Validation | $1,096.4M | 0.9827 | 0.0335 | 77.8% | 1.09% | 4.70% |
| Test | $1,358.3M | 0.9826 | 0.0339 | 77.7% | 1.11% | 4.67% |

The decision cutoff is `0.1488`.

- Approve the loan when the default probability is less than or equal to the cutoff.
- Deny the loan when the default probability is more than the cutoff.

## Profit Rule

The project uses this retrospective profit rule:

| Decision | Actual outcome | Profit value |
|---|---|---:|
| Approve | Paid in full | `+5% × DisbursementGross` |
| Approve | Default | `-25% × DisbursementGross` |
| Deny | Paid in full or default | `$0` |

The profit calculation uses `DisbursementGross`. The model does not use this field as a predictor.

## Model Scope

The workflow compares these model families:

| Model family | Models |
|---|---|
| Linear and discriminant | Ridge logistic regression, LDA, and QDA |
| Tree ensemble | Random Forest, Bagging, AdaBoost, and HistGradientBoosting |
| Boosting | XGBoost and LightGBM |
| Neural network | Multilayer perceptron |
| Model tuning | Optuna with a LightGBM model |

The final input has 37 numeric features and 7 categorical features. One-hot encoding makes 183 model columns.

## External Data

The workflow uses external data only when the data was available at or before the approval period.

| Data source | Data purpose |
|---|---|
| BLS unemployment | State labor market condition |
| BLS QCEW | State and industry employment conditions |
| FHFA HPI | Housing market condition |
| FEMA | Disaster exposure |
| Census CBP | Local industry condition |
| Engineered interactions | Combined economic and borrower conditions |

External features increased validation profit from `$1,087.9M` to `$1,096.4M`. The increase was approximately `$8.5M`.

## Data Leakage Controls

The model excludes outcome data and post-approval data.

| Excluded field | Reason |
|---|---|
| `MIS_Status` | Contains the target outcome |
| `ChgOffDate` | Occurs after a charge-off |
| `ChgOffPrinGr` | Occurs after a charge-off |
| `BalanceGross` | Contains post-approval balance data |
| `DisbursementGross` | Supplies only the retrospective profit value |

The workflow keeps the test set separate during model selection. The validation set selects the decision cutoff.

## Repository Contents

| Path | Content |
|---|---|
| `Main Workflow.py` | Main analysis and model workflow |
| `notebooks/final_workflow.ipynb` | Final notebook and selected model |
| `notebooks/README.md` | Notebook operating information |
| `reports/` | Final paper, presentation, and LaTeX source package |
| `reports/README.md` | Report package information |
| `COMPLETE_MODEL_RANKING.md` | Model results and robustness checks |
| `CONTRIBUTION_STATEMENT.md` | Team contribution record |

The repository excludes raw data, external-data caches, temporary outputs, and LaTeX build files.

## Operating Procedure

### Run the final notebook in Google Colab

1. Open `notebooks/final_workflow.ipynb` in Google Colab.

2. Read the instruction in each cell.

3. Run the cells in sequence.

4. Run the enriched dataset download cell.

5. Stop if a cell gives an error.

6. Correct the error before you continue.

7. Keep the test set protected until you freeze the model and cutoff.

### Run the main workflow on a local computer

1. Make a Python 3 environment.

2. Install the necessary packages.

   ```text
   pip install numpy pandas matplotlib scikit-learn lightgbm xgboost optuna pyarrow
   ```

3. Put an enriched dataset in a permitted local path.

4. Use `research_outputs/sba_enriched_eda_dataset.parquet` when this path is available.

5. Open `Main Workflow.py`.

6. Review each `RUN_...` control.

7. Set only the necessary controls to `True`.

8. Run the workflow by section.

9. Do not run the final test during model selection.

The repository does not supply the enriched dataset or a locked package environment.

## Final Deliverables

- [Research paper](reports/Group%202%20ML%20Final%20Project%20Research%20Paper.pdf)
- [Presentation](reports/Group%202%20ML%20Final%20Project%20Presentation.pdf)
- [LaTeX source package](reports/Group2_LaTeX_Source_Package_20260722.zip)
- [Complete model ranking](COMPLETE_MODEL_RANKING.md)
- [Contribution statement](CONTRIBUTION_STATEMENT.md)

## Writing Standard

The repository documentation applies the writing principles from [ASD-STE100 Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf).

The documentation uses controlled project terms, short sentences, active voice, and one action in each procedure step.

An approved STE checker did not certify the documentation.
