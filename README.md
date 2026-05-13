# SBA Loan Default Modeling

This project builds a loan approval decision workflow using default probability and validation net profit.

## Dataset

`SBAnational.csv` is required locally to run the workflow, but it is not included in GitHub because it is a data file.

## Files

- `Final Project Teaching.py` = learning/demo file
- `Main Workflow.py` = lab-style team workflow template
- `Teaching slide.pdf` = team teaching deck

## How to Run

1. Put `SBAnational.csv` in the project folder locally.
2. Open `Main Workflow.py` in Spyder, VS Code, or your preferred IDE.
3. Run setup blocks first.
4. Each teammate works only in their assigned sections.
5. Do not run final test until the team freezes features, model, hyperparameters, and threshold.

## Team Workflow

- Everyone does Part 1 EDA.
- Each teammate owns their assigned Part 2 model family.
- Private EDA can use any variable names.
- Shared model features must be created in `df_model` and registered in `numeric_cols` or `categorical_cols`.

## Git Rules

- `main` branch is stable.
- Teammates should work on their own branches.
- Open pull requests before merging.
- Do not commit data files or output CSVs.
