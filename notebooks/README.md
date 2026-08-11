# Final Notebook

This directory contains `final_workflow.ipynb`.

The notebook contains the final analysis, model comparison, selected model, and result figures.

## Operating Procedure

1. Open `final_workflow.ipynb` in Google Colab.

2. Read the instruction in each cell.

3. Run the cells in sequence.

4. Run the enriched dataset download cell.

5. Stop if a cell gives an error.

6. Correct the error before you continue.

7. Keep the test set protected during model selection.

8. Run the final test after you freeze the model and decision cutoff.

## Data and Environment

The notebook downloads `sba_enriched_eda_dataset.parquet` to the Colab runtime.

The notebook uses Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, Optuna, Matplotlib, and SHAP.

Package updates can change a result. Record the package versions before a new final test.

## Result Check

The selected model uses 44 input features. The decision cutoff is `0.1488` after rounding.

The validation net profit is `$1,096.4M`. The test net profit is `$1,358.3M`.

Compare a new result with `COMPLETE_MODEL_RANKING.md` before you accept it.
