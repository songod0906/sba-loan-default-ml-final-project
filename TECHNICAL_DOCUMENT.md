# SBA Loan Default Prediction

## Technical Description and Operating Instructions

| Item | Value |
|---|---|
| Document identifier | SBA-ML-TD-001 |
| Revision | 1.0 |
| Date | 2026-08-12 |
| Project type | Machine learning course project |
| Writing basis | ASD-STE100 Simplified Technical English, Issue 9 |

## 1. About this document

This document gives the technical description of the SBA loan default prediction project.

This document also gives the instructions to examine and run the project.

The author applied ASD-STE100 rules manually. An approved STE checker did not examine this document.

Machine learning terms are technical nouns in this document. Section 3 gives the approved project terms.

## 2. Important use limits

**IMPORTANT:** Do not use this model for a real loan decision.

This model is a course project and a portfolio artifact. It is not a production lending system.

A production system must have these controls:

- A legal and compliance review
- A fairness test
- An adverse-action explanation
- Model drift monitoring
- Data quality monitoring
- A manual review for a borderline decision

The project results show historical test performance. They do not give a guarantee of future performance.

## 3. Project terms

Use the same technical noun for the same item.

| Technical noun | Meaning |
|---|---|
| dataset | The table that contains one record for each SBA loan |
| development set | The data for analysis, feature design, training, and validation |
| training set | The part of the development set that fits the model |
| validation set | The part that selects the model and the decision cutoff |
| test set | The protected data that gives the final performance test |
| target | The loan outcome, where `0` is paid in full and `1` is default |
| default probability | The model estimate of the probability of default |
| decision cutoff | The maximum default probability for an approval |
| approval | The model decision to accept a loan in the historical test |
| net profit | The total test value from the project profit rule |
| external feature | A feature from an economic or disaster data source |
| data leakage | The use of data that was not available at the approval time |

## 4. System purpose

The system estimates the default probability for an SBA loan.

The system then compares this probability with the decision cutoff.

The business objective is maximum validation net profit. AUC is a secondary performance measure.

The final model is an Optuna-tuned LightGBM classifier. It uses 44 pre-approval features.

## 5. System sequence

| Sequence | System function | Output |
|---:|---|---|
| 1 | Load the enriched dataset | Full dataset |
| 2 | Make the development set and the protected test set | Two data parts |
| 3 | Make the training set and the validation set | Two development parts |
| 4 | Make and preprocess the model features | Model matrix |
| 5 | Fit the candidate models | Default probabilities |
| 6 | Select the decision cutoff on the validation set | Frozen decision cutoff |
| 7 | Apply the frozen model to the test set | Final test results |

## 6. Repository contents

| Path | Function |
|---|---|
| `README.md` | Gives the project summary |
| `TECHNICAL_DOCUMENT.md` | Gives the technical description and operating instructions |
| `Main Workflow.py` | Contains the main analysis and model workflow |
| `notebooks/final_workflow.ipynb` | Contains the final review workflow and selected model |
| `COMPLETE_MODEL_RANKING.md` | Gives the canonical model results |
| `reports/` | Contains the final paper, presentation, and LaTeX source package |
| `CONTRIBUTION_STATEMENT.md` | Gives the team contribution record |

The repository does not contain the raw dataset. It also does not contain the external data cache.

## 7. Operating environment

Use a Python 3 environment. You can use a local computer or Google Colab.

The core workflow imports these packages:

- NumPy
- pandas
- Matplotlib
- scikit-learn

The model comparison can also use these packages:

- LightGBM
- XGBoost
- Optuna
- PyArrow for a Parquet dataset

The repository does not supply a locked environment file. Package updates can change a result.

## 8. Input data

### 8.1 Required dataset

The main workflow requires an enriched dataset. Put one permitted file in the project directory.

The workflow searches for these Parquet files:

1. `research_outputs/sba_enriched_eda_dataset.parquet`
2. `sba_enriched_eda_dataset.parquet`
3. `research_outputs/haianh_improved_eda_dataset.parquet`
4. `haianh_improved_eda_dataset.parquet`

The workflow also searches for these CSV files:

1. `research_outputs/sba_enriched_eda_dataset.csv`
2. `sba_enriched_eda_dataset.csv`

The public repository does not contain these files. The final LaTeX package contains result tables, but it does not replace the dataset.

### 8.2 Target and profit amount

The target column is `y`.

- `y = 0` means that the loan was paid in full.
- `y = 1` means that the loan defaulted.

The profit calculation uses `DisbursementGross`. If this column is not available, the workflow uses `GrAppv`.

The model does not use `DisbursementGross` as a predictor.

### 8.3 Split controls

The final analysis used these split sizes:

| Data part | Records | Function |
|---|---:|---|
| Training set | 554,292 | Fit the model |
| Validation set | 138,573 | Select the model and decision cutoff |
| Test set | 173,217 | Give the final test result |

The workflow uses random state `1`. It tries to preserve the target and era distributions.

Do not use the test set during model selection.

## 9. Data leakage controls

The workflow removes outcome data and post-approval data from the model matrix.

| Removed field | Reason |
|---|---|
| `MIS_Status` | Contains the target outcome |
| `ChgOffDate` | Occurs after a charge-off |
| `ChgOffPrinGr` | Occurs after a charge-off |
| `BalanceGross` | Contains post-approval balance data |
| `DisbursementGross` | Is only for the retrospective profit calculation |

External features must contain data that was available at or before the approval period.

## 10. Feature set

The final feature set contains 37 numeric features and 7 categorical features.

One-hot encoding makes 183 model columns. The source-of-truth file reports 44 input features.

| Feature group | Examples |
|---|---|
| SBA loan data | Term, employee count, approval amount, guarantee amount |
| Borrower data | Business age, industry, state, urban or rural class |
| Lender data | Bank state and same-state lender indicator |
| Economic data | Inflation, oil price, income, employment, and house price data |
| Disaster data | Hurricane, fire, flood, and severe-storm data |
| Interaction data | Industry, housing, disaster, and small-business risk combinations |

The workflow imputes a missing numeric value with the median. It one-hot encodes a categorical value.

The linear models and the neural network use scaled numeric values. The tree models use unscaled numeric values.

## 11. Selected model

The final model is `LGB Optuna (n418 nl194)`.

| Parameter | Value |
|---|---:|
| `n_estimators` | 418 |
| `num_leaves` | 194 |
| `learning_rate` | 0.0637 |
| `min_child_samples` | 85 |
| `subsample` | 0.50 |
| `colsample_bytree` | 0.82 |
| `reg_alpha` | 0.0011 |
| `reg_lambda` | 0.017 |
| `random_state` | 1 |

The final notebook contains these parameters. The main script contains the wider team workflow.

## 12. Decision and profit rules

Approve a loan when the default probability is less than or equal to `0.1488`.

Deny a loan when the default probability is more than `0.1488`.

The project uses this retrospective profit rule:

| Decision | Actual outcome | Profit value |
|---|---|---:|
| Approve | Paid in full | `+0.05 × DisbursementGross` |
| Approve | Default | `-0.25 × DisbursementGross` |
| Deny | Paid in full or default | `0` |

The validation set selects the decision cutoff. The test set does not change the cutoff.

## 13. Operating procedure

### 13.1 Get the project

1. Clone the repository.

   ```text
   git clone https://github.com/songod0906/sba-loan-default-ml-final-project.git
   ```

2. Go to the project directory.

3. Make a Python 3 environment.

4. Install the necessary packages.

   ```text
   pip install numpy pandas matplotlib scikit-learn lightgbm xgboost optuna pyarrow jupyter
   ```

### 13.2 Prepare the data

1. Get the authorized SBA data.

2. Make the enriched dataset outside the public repository.

3. Put the enriched dataset in one of the paths in Section 8.1.

4. Make sure that the dataset contains the target and required feature columns.

5. Do not commit the dataset to the repository.

### 13.3 Run the final notebook

1. Open `notebooks/final_workflow.ipynb`.

2. Read the instruction in each cell.

3. Run the cells in sequence.

4. Stop if a cell gives an error.

5. Correct the error before you continue.

6. Keep the test set protected until the model and cutoff are frozen.

7. Run the final test one time.

### 13.4 Run the main script

1. Open `Main Workflow.py`.

2. Make sure that the dataset path is correct.

3. Review each `RUN_...` control before you change it.

4. Set only the necessary model controls to `True`.

5. Run the script by section when your editor supports sections.

6. Do not set `RUN_FINAL_TEST` to `True` during model selection.

The main script writes generated files to `research_outputs/final_workflow_outputs/`.

## 14. Reference results

Use these values to examine a rerun:

| Measure | Validation | Test |
|---|---:|---:|
| Net profit | $1,096.4 million | $1,358.3 million |
| AUC | 0.9827 | 0.9826 |
| Brier score | 0.0335 | 0.0339 |
| Approval rate | 77.8% | 77.7% |
| Approved default rate | 1.09% | 1.11% |
| ROI | 4.70% | 4.67% |

A small difference can occur because of package, hardware, or dataset differences.

Stop the examination if the split sizes, feature count, or decision cutoff are different.

## 15. Verification procedure

1. Make sure that the split sizes agree with Section 8.3.

2. Make sure that the input feature count is 44.

3. Make sure that the decision cutoff is `0.1488` after rounding.

4. Compare the new validation results with Section 14.

5. Compare the new test results only after the final test procedure.

6. Record the Python and package versions.

7. Record each data source and its retrieval date.

8. Keep the new result files outside the Git repository until you examine them.

## 16. Troubleshooting

| Condition | Possible cause | Corrective action |
|---|---|---|
| The workflow shows `Dataset: NOT FOUND` | The dataset is not in an approved path | Put the dataset in a path from Section 8.1 |
| Parquet loading fails | PyArrow is not installed or the file has a defect | Install PyArrow or use the CSV file |
| LightGBM does not run | LightGBM is not installed | Install LightGBM and run the import cell again |
| A model section does not run | Its `RUN_...` control is `False` | Set only the necessary control to `True` |
| The result is different | The data, split, package, or parameters changed | Compare the configuration with this document |
| The computer has insufficient memory | The full dataset and encoded matrix are large | Use a system with more memory or use Google Colab |

## 17. Known limits

- The repository does not contain the dataset.
- The repository does not contain a locked package environment.
- The main script requires an enriched dataset that a separate data process makes.
- The final model is not an application programming interface.
- The project does not include a production monitoring service.
- The project does not complete a legal, fairness, or adverse-action review.
- The profit rule is a project assumption, not an audited lending policy.

## 18. Source-of-truth files

Use these files when two project values are different:

1. `reports/Group2_LaTeX_Source_Package_20260722.zip`
2. `submission_latex_package_20260722/paper/tables/PAPER_TRUTH.json` inside the ZIP file
3. `COMPLETE_MODEL_RANKING.md`
4. `notebooks/final_workflow.ipynb`

## 19. ASD-STE100 reference

This document uses the principles in [ASD-STE100 Simplified Technical English, Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf).

ASD owns the ASD-STE100 standard. This project does not reproduce the standard or its dictionary.
