# -*- coding: utf-8 -*-
"""
SBA Final Project — Main Workflow Lab Template

How to use it:
1. Run the early setup blocks together.
2. Use the EDA and feature blocks to explore your own ideas.
3. When a feature becomes part of the shared model, add it to df_model and register it in
   numeric_cols or categorical_cols.
4. Each teammate works inside their assigned model block.
5. We compare everyone with the same validation scoreboard.
6. Set model = True to run, for example:
    RUN_TREE_MODELS = True

Important notes:
- Your private EDA variables can have any names.
- The shared model only reads these names:
    df_model
    y
    amount
    numeric_cols
    categorical_cols
    X_train, X_valid, X_test
    y_train, y_valid, y_test
    amount_train, amount_valid, amount_test
    preprocess_scaled, preprocess_tree

That means you have freedom while exploring, but our final model still follows the rules above.

Dataset expected in this folder:
SBAnational.csv
"""

#%% 1. Import libraries

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

RANDOM_STATE = 1
DATA_PATH = "SBAnational.csv"

# Keep this True while learning / debugging.
# For the final full run, change it to False.
USE_WORKING_SAMPLE = True
WORKING_SAMPLE_N = 50000

THEORETICAL_DEFAULT_THRESHOLD = 1 / 6

# Fill your name here when you run your own copy.
# This is just for notes / saved outputs. It does not affect the model.
OWNER_NAME = ""


#%% 2. Load data

df_raw = pd.read_csv(DATA_PATH)

print("Rows, columns:", df_raw.shape)
print(df_raw.head())
print(df_raw.columns)


#%% 3. Create target and clean money columns

# y = 1 means default / charged off.
# y = 0 means paid in full.

# Always restart from df_raw so rerunning cells does not stack changes.
df = df_raw.copy()

print("MIS_Status distribution before cleaning:")
print(df["MIS_Status"].value_counts(dropna=False))

# Remove rows with missing target.
df = df[df["MIS_Status"].notna()].copy()

df["y"] = np.where(df["MIS_Status"] == "CHGOFF", 1, 0)

print("Target distribution:")
print(df["y"].value_counts())
print("Default rate:", df["y"].mean())

# Money columns are stored like "$60,000.00". Convert them to float.
money_cols = ["DisbursementGross", "GrAppv", "SBA_Appv", "BalanceGross", "ChgOffPrinGr"]

for col in money_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"nan": np.nan, "": np.nan})
            .astype(float)
        )

print(df[[c for c in money_cols if c in df.columns]].head())


#%% 4. Optional working sample

# This keeps the file fast while we are learning.
# Stratify keeps the default/non-default ratio similar.

if USE_WORKING_SAMPLE:
    sample_n = min(WORKING_SAMPLE_N, len(df))

    if sample_n < len(df):
        df, _ = train_test_split(
            df,
            train_size=sample_n,
            random_state=RANDOM_STATE,
            stratify=df["y"],
        )

    df = df.sort_index().copy()
    print("Using working sample:", df.shape)
    print("Sample default rate:", df["y"].mean())
else:
    print("Using full dataset:", df.shape)


#%% 5. Leakage audit and modeling dataframe

# Question:
# Which columns would not be known when the bank is deciding whether to approve the loan?
# Remove those columns from X.

leakage_cols = [
    "MIS_Status",
    "ChgOffDate",
    "ChgOffPrinGr",
    "BalanceGross",
]

id_text_cols = [
    "LoanNr_ChkDgt",
    "Name",
    "City",
    "Zip",
    "Bank",
]

# DisbursementGross is used for profit calculation.
# Decide separately whether it should be used as a predictor. What do you think, explain your decision
USE_DISBURSEMENT_AS_PREDICTOR = False

df_model = df.drop(columns=leakage_cols + id_text_cols, errors="ignore").copy()

print("Model dataframe shape:", df_model.shape)
print(df_model.columns)

# Answer / notes:
#
#


#%% 6. Part 1 EDA workspace — everyone does this first

# Rule:
# Use any variable names you want here.
# Your private EDA code will not affect the shared model unless you later create a column in df_model
# and register it in numeric_cols or categorical_cols.

# Required Part 1 output for everyone:
# 1. Three EDA observations
# 2. Two possible feature ideas
# 3. One leakage concern
# 4. Table and Visualization
# 5. One short interpretation for the bank. 


#%% 6A. Part 1 EDA — Owner: Hai An

# Questions to answer:
# - Which variables look useful before modeling?
# - Which feature ideas do you want to try?
# - Is there any leakage risk in your ideas?

# Write your EDA code here.
# Example only:
# hai_an_state_risk = df_model.groupby("State")["y"].agg(["count", "mean"]).sort_values("mean", ascending=False)
# print(hai_an_state_risk.head(15))

# Hai An notes:
# 1. EDA observation:
# 2. EDA observation:
# 3. EDA observation:
# Feature ideas:
# Leakage concern:
# Lending interpretation:


#%% 6B. Part 1 EDA — Owner: Hai Anh

# Write your EDA code here.
# Focus idea: variables that may help logistic regression, LDA, or QDA.
# Example only:
# hai_anh_term_summary = df_model.groupby("y")["Term"].median()
# print(hai_anh_term_summary)

# Hai Anh notes:
# 1. EDA observation:
# 2. EDA observation:
# 3. EDA observation:
# Feature ideas:
# Leakage concern:
# Lending interpretation:


#%% 6C. Part 1 EDA — Owner: Huyen Anh

# Write your EDA code here.
# Focus idea: patterns that may require nonlinear models or interactions.
# Example only:
# huyen_anh_revline = df_model.groupby("RevLineCr")["y"].agg(["count", "mean"]).sort_values("mean", ascending=False)
# print(huyen_anh_revline.head(15))

# Huyen Anh notes:
# 1. EDA observation:
# 2. EDA observation:
# 3. EDA observation:
# Feature ideas:
# Leakage concern:
# Lending interpretation:


#%% 7. Shared feature engineering workspace

# Rule:
# If your feature should be used by the team, create it here with a stable column name.
# Then register the column in Block 8.

# Starter features from our teaching file are listed below.
# They are comments on purpose. Choose necessary features based on your reasoning. 

# 7.1 SBA guarantee portion
# df_model["Portion"] = df_model["SBA_Appv"] / df_model["GrAppv"]
# df_model["Portion"] = df_model["Portion"].replace([np.inf, -np.inf], np.nan).fillna(0)
# df_model["unguaranteed_ratio"] = 1 - df_model["Portion"]
# df_model["unguaranteed_amount"] = df_model["GrAppv"] - df_model["SBA_Appv"]

# 7.2 Job impact
# df_model["jobs_total"] = df_model["CreateJob"] + df_model["RetainedJob"]
# df_model["jobs_per_dollar"] = df_model["jobs_total"] / df_model["GrAppv"]
# df_model["jobs_per_dollar"] = df_model["jobs_per_dollar"].replace([np.inf, -np.inf], np.nan).fillna(0)

# 7.3 Same-state lender
# df_model["same_state_bank"] = np.where(df_model["State"] == df_model["BankState"], 1, 0)

# 7.4 Real-estate proxy
# df_model["RealEstate"] = np.where(df_model["Term"] >= 240, 1, 0)

# 7.5 Date / recession features
# for col in ["ApprovalDate", "DisbursementDate"]:
#     df_model[col] = pd.to_datetime(df_model[col], errors="coerce")
#
# recession_start = pd.Timestamp("2007-12-01")
# recession_end = pd.Timestamp("2009-06-30")
# df_model["estimated_maturity_date"] = df_model["DisbursementDate"] + pd.to_timedelta(df_model["Term"].fillna(0) * 30, unit="D")
# df_model["Recession"] = np.where(
#     (df_model["DisbursementDate"] <= recession_end) &
#     (df_model["estimated_maturity_date"] >= recession_start),
#     1,
#     0,
# )
# df_model["approval_year"] = df_model["ApprovalDate"].dt.year
# df_model["disbursement_year"] = df_model["DisbursementDate"].dt.year

# 7.6 NAICS sector
# df_model["NAICS_str"] = df_model["NAICS"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
# df_model["NAICS_sector"] = df_model["NAICS_str"].str[:2]

# 7.7 Clean LowDoc and RevLineCr
# def clean_yes_no(series):
#     cleaned = series.astype(str).str.strip().str.upper()
#     cleaned = cleaned.replace({
#         "Y": "Yes", "YES": "Yes", "1": "Yes",
#         "N": "No", "NO": "No", "0": "No",
#         "NAN": "Unknown", "": "Unknown",
#     })
#     cleaned = np.where(pd.Series(cleaned).isin(["Yes", "No"]), cleaned, "Unknown")
#     return cleaned
#
# df_model["LowDoc_clean"] = clean_yes_no(df_model["LowDoc"])
# df_model["RevLineCr_clean"] = clean_yes_no(df_model["RevLineCr"])

# 7.8 Log transforms
# for col in ["GrAppv", "SBA_Appv", "DisbursementGross", "NoEmp"]:
#     if col in df_model.columns:
#         df_model[f"log_{col}"] = np.log1p(df_model[col])

# 7.9 Interaction ideas
# df_model["RealEstate_x_Portion"] = df_model["RealEstate"] * df_model["Portion"]
# df_model["Recession_x_Portion"] = df_model["Recession"] * df_model["Portion"]
# df_model["Recession_x_RealEstate"] = df_model["Recession"] * df_model["RealEstate"]

# Add your own feature ideas below:
#
#

# Quick check after you create features:
print("Current df_model columns:")
print(df_model.columns)


#%% 8. Register predictors

# This is the shared lists.
# If a feature is not listed here, the model will not use it.

numeric_cols = [
    # Basic SBA numeric fields
    "Term",
    "NoEmp",
    "CreateJob",
    "RetainedJob",
    "GrAppv",
    "SBA_Appv",

    # Add engineered numeric features here after creating them in Block 7
    # "Portion",
    # "unguaranteed_ratio",
    # "unguaranteed_amount",
    # "jobs_total",
    # "jobs_per_dollar",
    # "same_state_bank",
    # "RealEstate",
    # "Recession",
    # "approval_year",
    # "disbursement_year",
    # "log_GrAppv",
    # "log_SBA_Appv",
    # "log_NoEmp",
    # "RealEstate_x_Portion",
    # "Recession_x_Portion",
    # "Recession_x_RealEstate",
]

if USE_DISBURSEMENT_AS_PREDICTOR:
    numeric_cols += ["DisbursementGross"]

categorical_cols = [
    "State",
    "BankState",
    "NewExist",
    "UrbanRural",

    # Add engineered categorical features here after creating them in Block 7
    # "NAICS_sector",
    # "LowDoc_clean",
    # "RevLineCr_clean",
]

# Keep only columns that actually exist.
numeric_cols = [c for c in numeric_cols if c in df_model.columns]
categorical_cols = [c for c in categorical_cols if c in df_model.columns]

print("Numeric predictors:", numeric_cols)
print("Categorical predictors:", categorical_cols)

# Answer / notes:
# Why did you choose these predictors?
#
#


#%% 9. Build X, y, and amount

X = df_model[numeric_cols + categorical_cols].copy()
y = df["y"].copy()
amount = df["DisbursementGross"].copy()

# Convert categorical columns to string before one-hot encoding.
for col in categorical_cols:
    X[col] = X[col].fillna("Unknown").astype(str)

print("X shape:", X.shape)
print("y default rate:", y.mean())
print("Amount missing values:", amount.isna().sum())


#%% 10. Train / validation / test split

# Use 60% train, 20% validation, 20% test.
# Validation is where we compare models and tune thresholds.
# Test stays untouched until the end.

X_train_valid, X_test, y_train_valid, y_test, amount_train_valid, amount_test = train_test_split(
    X,
    y,
    amount,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train, X_valid, y_train, y_valid, amount_train, amount_valid = train_test_split(
    X_train_valid,
    y_train_valid,
    amount_train_valid,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_train_valid,
)

print("Train:", X_train.shape, "default rate:", y_train.mean())
print("Valid:", X_valid.shape, "default rate:", y_valid.mean())
print("Test:", X_test.shape, "default rate:", y_test.mean())


#%% 11. Preprocessing objects

# Use preprocess_scaled for KNN, logistic, NN, LDA/QDA.
# Use preprocess_tree for tree, bagging, RF, boosting.

numeric_transformer_scaled = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

numeric_transformer_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
])

preprocess_scaled = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_scaled, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

preprocess_tree = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_tree, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)


#%% 12. Shared scoreboard functions

# These functions are provided so everyone uses the same scoring rule.
# You should still understand them from the teaching file.


def loan_profit_vector(y_true, decision_approve, amount_series):
    """Return profit/loss for each loan."""
    amount_arr = np.asarray(amount_series, dtype=float)
    y_arr = np.asarray(y_true)
    approve_arr = np.asarray(decision_approve).astype(bool)

    gain_if_paid = 0.05 * amount_arr
    loss_if_default = -0.25 * amount_arr

    return np.where(
        approve_arr,
        np.where(y_arr == 0, gain_if_paid, loss_if_default),
        0.0,
    )


def evaluate_prob_model(model_name, y_true, prob_default, amount_series, threshold):
    """Evaluate one model at one default-probability threshold."""
    y_true_arr = np.asarray(y_true)
    prob_default_arr = np.asarray(prob_default)

    pred_default = (prob_default_arr > threshold).astype(int)
    decision_approve = prob_default_arr <= threshold

    confmat = confusion_matrix(y_true_arr, pred_default, labels=[1, 0])

    TP = confmat[0, 0]
    FN = confmat[0, 1]
    FP = confmat[1, 0]
    TN = confmat[1, 1]

    profit = loan_profit_vector(y_true_arr, decision_approve, amount_series)

    return {
        "model": model_name,
        "threshold_default": threshold,
        "threshold_success": 1 - threshold,
        "accuracy": accuracy_score(y_true_arr, pred_default),
        "recall_default": recall_score(y_true_arr, pred_default, zero_division=0),
        "precision_default": precision_score(y_true_arr, pred_default, zero_division=0),
        "specificity_paid": TN / (TN + FP) if (TN + FP) > 0 else np.nan,
        "f1_default": f1_score(y_true_arr, pred_default, zero_division=0),
        "auc": roc_auc_score(y_true_arr, prob_default_arr),
        "brier": brier_score_loss(y_true_arr, prob_default_arr),
        "net_profit": profit.sum(),
        "approval_rate": decision_approve.mean(),
        "approved_default_rate": y_true_arr[decision_approve].mean() if decision_approve.sum() > 0 else np.nan,
        "denied_default_rate": y_true_arr[~decision_approve].mean() if (~decision_approve).sum() > 0 else np.nan,
        "TP_default_denied": TP,
        "FN_default_approved": FN,
        "FP_paid_denied": FP,
        "TN_paid_approved": TN,
    }


def tune_threshold_by_profit(model_name, y_true, prob_default, amount_series, thresholds=None):
    """Search validation thresholds and sort by net profit."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 120)

    thresholds = np.unique(np.append(thresholds, THEORETICAL_DEFAULT_THRESHOLD))

    rows = []
    for th in thresholds:
        rows.append(evaluate_prob_model(model_name, y_true, prob_default, amount_series, th))

    return pd.DataFrame(rows).sort_values("net_profit", ascending=False).reset_index(drop=True)


def fit_predict_evaluate(model_name, estimator, preprocess_obj):
    """Fit one model and return validation results."""
    start = time.perf_counter()

    pipe = Pipeline(steps=[
        ("preprocess", preprocess_obj),
        ("model", estimator),
    ])

    pipe.fit(X_train, y_train)
    prob_default_valid = pipe.predict_proba(X_valid)[:, 1]

    runtime = time.perf_counter() - start

    theory = evaluate_prob_model(
        model_name,
        y_valid,
        prob_default_valid,
        amount_valid,
        THEORETICAL_DEFAULT_THRESHOLD,
    )
    theory["threshold_type"] = "theoretical_1_over_6"
    theory["runtime_seconds"] = runtime

    threshold_table = tune_threshold_by_profit(
        model_name,
        y_valid,
        prob_default_valid,
        amount_valid,
    )

    tuned = threshold_table.iloc[0].to_dict()
    tuned["threshold_type"] = "validation_profit_tuned"
    tuned["runtime_seconds"] = runtime

    return pipe, prob_default_valid, theory, tuned, threshold_table


#%% 13. Baseline policies

approve_all_profit = loan_profit_vector(
    y_true=y_valid,
    decision_approve=np.ones(len(y_valid), dtype=bool),
    amount_series=amount_valid,
).sum()

deny_all_profit = 0.0

print("Approve-all validation profit:", approve_all_profit)
print("Deny-all validation profit:", deny_all_profit)


#%% 14A. KNN workspace — Owner: Hai An

RUN_KNN = False

if RUN_KNN:
    # Questions:
    # 1. Which k values did you try first?
    # 2. Did weights="uniform" or weights="distance" work better?
    # 3. Which preprocessing object should KNN use?
    # 4. Does KNN beat approve-all profit?

    # Write your first KNN model here.
    # Keep the first run simple. Don't try to tune everything at once.
    #
    # Example start:
    # knn = KNeighborsClassifier(n_neighbors=___, weights="___", p=___)
    # knn_pipe, knn_prob, knn_theory, knn_tuned, knn_threshold_table = fit_predict_evaluate(
    #     "KNN", knn, preprocess_obj=preprocess_scaled
    # )
    # print(pd.DataFrame([knn_theory, knn_tuned]))
    # print(knn_threshold_table.head(10))
    pass

# Hai An KNN notes:
# First run result:
# Best validation threshold:
# Validation net profit:
# Approved default rate:
# Interpretation:


#%% 14B. Decision tree / bagging / random forest / boosting workspace — Owner: Hai An

RUN_TREE_MODELS = False

if RUN_TREE_MODELS:
    # Questions:
    # 1. For one tree, what max_depth and min_samples_leaf did you try?
    # 2. Does bagging improve over one tree?
    # 3. For Random Forest, does max_features="sqrt", 0.5, or None work better?
    # 4. Does boosting improve over bagging or Random Forest?

    # Write your tree-family models here.
    # Use preprocess_tree.
    # Keep each model result separately so we can compare them later.
    #
    # Example result names you can use:
    # tree_tuned, bag_tuned, rf_tuned, boost_tuned
    pass

# Hai An tree-family notes:
# Best tree result:
# Best bagging result:
# Best RF result:
# Best boosting result:
# Interpretation:


#%% 14C. Logistic regression and discriminant analysis workspace — Owner: Hai Anh

RUN_LOGIT_DA = False

if RUN_LOGIT_DA:
    # Questions:
    # 1. Try Ridge, Lasso, and ElasticNet.
    # 2. Which C values did you try?
    # 3. Did the solver converge? Check n_iter_.
    # 4. For LDA/QDA, did regularization help?

    # Write your logistic / discriminant models here.
    # Use preprocess_scaled.
    #
    # Example result names you can use:
    # ridge_tuned, lasso_tuned, elastic_tuned, lda_tuned, qda_best
    pass

# Hai Anh notes:
# Best logistic result:
# Best LDA/QDA result:
# Solver/convergence concern:
# Interpretation:


#%% 14D. Neural network workspace — Owner: Huyen Anh

RUN_NEURAL_NET = False

if RUN_NEURAL_NET:
    # Questions:
    # 1. What hidden_layer_sizes did you try?
    # 2. Which activation did you use?
    # 3. Does the loss curve fall smoothly?
    # 4. Does higher flexibility improve validation net profit?

    # Write your neural network models here.
    # Use preprocess_scaled.
    #
    # Example result names you can use:
    # mlp_tuned, mlp_threshold_table
    pass

# Huyen Anh notes:
# Best NN architecture:
# Best validation threshold:
# Validation net profit:
# Loss curve observation:
# Interpretation:


#%% 15. Cross-validation helper and result log

# Everyone should run CV / tuning for their own assigned model family.
# CV happens on X_train / y_train only.
# Do not use X_valid inside cross-validation.

cv_results = []

def add_cv_result(owner, model_family, params, scores):
    """Append one CV result row to the shared CV log."""
    cv_results.append({
        "owner": owner,
        "model_family": model_family,
        "params": params,
        "cv_auc_mean": float(np.mean(scores)),
        "cv_auc_std": float(np.std(scores)),
    })


#%% 15A. KNN cross-validation — Owner: Hai An

RUN_KNN_CV = False # Set to True to run

if RUN_KNN_CV:
    # Goal:
    # Try several KNN settings using CV on training data only.
    # After CV, choose a small number of promising settings and evaluate them on validation.

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Write your CV loop here.
    # Suggested grid:
    # k_values = [11, 31, 51, 101]
    # weights_options = ["uniform", "distance"]
    #
    # for k in k_values:
    #     for w in weights_options:
    #         candidate = Pipeline(steps=[
    #             ("preprocess", preprocess_scaled),
    #             ("model", KNeighborsClassifier(n_neighbors=k, weights=w, p=2, n_jobs=-1)),
    #         ])
    #         scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    #         add_cv_result("Hai An", "KNN", {"k": k, "weights": w}, scores)
    #         print(k, w, scores.mean())
    pass

# Hai An KNN CV notes:
# Best CV setting:
# Did CV agree with validation profit?


#%% 15B. Tree-family cross-validation — Owner: Hai An

RUN_TREE_CV = False # Set to True to run

if RUN_TREE_CV:
    # Goal:
    # Run a small CV grid for the strongest tree-family model.
    # Do not make the grid too large at first.

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Write your CV loop here.
    # Suggested Random Forest grid:
    # for depth in [12, 16]:
    #     for leaf in [50, 100]:
    #         for max_feat in [None, 0.5, "sqrt"]:
    #             candidate = Pipeline(steps=[
    #                 ("preprocess", preprocess_tree),
    #                 ("model", RandomForestClassifier(
    #                     n_estimators=50,
    #                     max_depth=depth,
    #                     min_samples_leaf=leaf,
    #                     max_features=max_feat,
    #                     n_jobs=1,
    #                     random_state=RANDOM_STATE,
    #                 )),
    #             ])
    #             scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    #             add_cv_result("Hai An", "Random Forest", {
    #                 "max_depth": depth,
    #                 "min_samples_leaf": leaf,
    #                 "max_features": max_feat,
    #             }, scores)
    #             print(depth, leaf, max_feat, scores.mean())
    pass

# Hai An tree CV notes:
# Best CV setting:
# Did max_features matter?


#%% 15C. Logistic / discriminant cross-validation — Owner: Hai Anh

RUN_LOGIT_DA_CV = False # Set to True to run

if RUN_LOGIT_DA_CV:
    # Goal:
    # Run CV for logistic settings.
    # LDA/QDA can be checked separately if memory becomes heavy.

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Write your CV loop here.
    # Suggested Ridge/Lasso grid:
    # for penalty, C, l1_ratio in [
    #     ("l2", 0.1, None),
    #     ("l2", 1.0, None),
    #     ("l1", 0.1, None),
    #     ("l1", 0.5, None),
    #     ("elasticnet", 0.5, 0.5),
    # ]:
    #     kwargs = {
    #         "penalty": penalty,
    #         "C": C,
    #         "solver": "saga",
    #         "max_iter": 3000,
    #         "n_jobs": -1,
    #         "random_state": RANDOM_STATE,
    #     }
    #     if penalty == "elasticnet":
    #         kwargs["l1_ratio"] = l1_ratio
    #     candidate = Pipeline(steps=[
    #         ("preprocess", preprocess_scaled),
    #         ("model", LogisticRegression(**kwargs)),
    #     ])
    #     scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    #     add_cv_result("Hai Anh", "Logistic", {"penalty": penalty, "C": C, "l1_ratio": l1_ratio}, scores)
    #     print(penalty, C, l1_ratio, scores.mean())
    pass

# Hai Anh CV notes:
# Best logistic CV setting:
# Did Lasso/ElasticNet improve over Ridge?


#%% 15D. Neural network cross-validation / tuning — Owner: Huyen Anh

RUN_NN_CV = False # Set to True to run

if RUN_NN_CV:
    # Goal:
    # Try a small neural network grid.
    # Keep it small first because NN can become slow.

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Write your CV loop here.
    # Suggested grid:
    # architectures = [(32,), (64, 32), (128, 64)]
    # alphas = [0.0001, 0.001]
    #
    # for arch in architectures:
    #     for alpha in alphas:
    #         candidate = Pipeline(steps=[
    #             ("preprocess", preprocess_scaled),
    #             ("model", MLPClassifier(
    #                 hidden_layer_sizes=arch,
    #                 activation="relu",
    #                 solver="adam",
    #                 alpha=alpha,
    #                 early_stopping=True,
    #                 validation_fraction=0.10,
    #                 max_iter=100,
    #                 random_state=RANDOM_STATE,
    #             )),
    #         ])
    #         scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    #         add_cv_result("Huyen Anh", "Neural Network", {"hidden_layer_sizes": arch, "alpha": alpha}, scores)
    #         print(arch, alpha, scores.mean())
    pass

# Huyen Anh CV notes:
# Best NN CV setting:
# Did larger networks help enough to justify runtime?


#%% 15E. CV summary table

if len(cv_results) > 0:
    cv_results_df = pd.DataFrame(cv_results).sort_values(
        ["owner", "cv_auc_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)

    print(cv_results_df)
else:
    print("No CV results yet. Each owner can turn on their own RUN_*_CV flag.")

# Answer / notes:
# Which settings are worth taking to validation?
#
#


#%% 16. Build validation leaderboard

# Each owner should append their best result dictionary into model_results.
# Usually use the validation-profit-tuned row, not the theoretical threshold row.

model_results = []

# Examples after running model blocks:
# model_results.append(knn_tuned)
# model_results.append(tree_tuned)
# model_results.append(rf_tuned)
# model_results.append(ridge_tuned)
# model_results.append(mlp_tuned)

baseline_rows = [
    {
        "model": "Approve All Baseline",
        "threshold_type": "baseline",
        "net_profit": approve_all_profit,
        "approval_rate": 1.0,
        "approved_default_rate": y_valid.mean(),
    },
    {
        "model": "Deny All Baseline",
        "threshold_type": "baseline",
        "net_profit": deny_all_profit,
        "approval_rate": 0.0,
        "approved_default_rate": np.nan,
    },
]

leaderboard = pd.DataFrame(baseline_rows + model_results)
leaderboard = leaderboard.sort_values("net_profit", ascending=False).reset_index(drop=True)

print(leaderboard)

# Answer / notes:
# Which model is strongest on validation profit?
#
#


#%% 17. Profit curve for finalist model

# After the team chooses one finalist, build the profit curve here.
# Do not do this before the model comparison is clear.

RUN_PROFIT_CURVE = False

if RUN_PROFIT_CURVE:
    # Fill these from the chosen model block.
    # finalist_name = "__________"
    # finalist_prob_default = __________

    # Sort loans from safest to riskiest.
    # Approve progressively.
    # Track cumulative profit.
    # Choose approval depth that maximizes validation profit.
    pass

# Answer / notes:
# Final validation policy: approve if P(default) <= ________
# Approval rate: ________
# Validation profit: ________


#%% 18. ROC, calibration, and error analysis

# These are diagnostic checks after choosing the finalist model.
# Keep them short and focused.

RUN_DIAGNOSTICS = False

if RUN_DIAGNOSTICS:
    # ROC: ranking quality.
    # Calibration: whether predicted probabilities are close to actual default rates.
    # Error analysis: where approved defaults and denied paid loans happen.
    pass

# Answer / notes:
#
#


#%% 19. Final untouched test evaluation

RUN_FINAL_TEST = False

if RUN_FINAL_TEST:
    # Only run this after the team freezes:
    # - feature list
    # - preprocessing choice
    # - model family
    # - hyperparameters
    # - threshold rule

    # Refit finalist model on train + validation.
    # Evaluate once on test.
    # Do not tune after seeing test result.
    pass

# Answer / notes:
# Final test result:
#
#


#%% 20. Save outputs

# Save only after the team has real results.

# leaderboard.to_csv("sba_validation_leaderboard.csv", index=False)
# Add other outputs here:
# - profit curve table
# - calibration table
# - final test result

print("Workflow template finished. Fill model blocks, then save outputs.")
