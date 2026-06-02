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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
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
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
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
OWNER_NAME = "HAI AN"


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

# ------------------------------------------------------------
# 1. Target distribution
# ------------------------------------------------------------

print("\n1. TARGET DISTRIBUTION")
target_table = df_model["y"].value_counts().rename(index={0: "Paid in Full", 1: "Default / Charged Off"})
target_rate = df_model["y"].value_counts(normalize=True).rename(index={0: "Paid in Full", 1: "Default / Charged Off"})

target_summary = pd.DataFrame({
    "count": target_table,
    "percentage": target_rate
})

print(target_summary)

# Visualization: target distribution
plt.figure(figsize=(6, 4))
target_summary["percentage"].plot(kind="bar", edgecolor="black")
plt.title("Loan Outcome Distribution")
plt.ylabel("Percentage")
plt.xlabel("Loan Status")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 2. EDA observation 1: Term and default
# ------------------------------------------------------------

print("\n2. DEFAULT RATE BY TERM GROUP")

df_model["term_group_temp"] = pd.cut(
    df_model["Term"],
    bins=[0, 36, 84, 120, 240, np.inf],
    labels=["<=36 months", "37-84 months", "85-120 months", "121-240 months", ">240 months"]
)

term_risk = df_model.groupby("term_group_temp")["y"].agg(["count", "mean"]).reset_index()
term_risk.columns = ["Term Group", "Loan Count", "Default Rate"]
term_risk["Default Rate"] = term_risk["Default Rate"].round(4)

print(term_risk)

# Visualization: default rate by term group
plt.figure(figsize=(8, 5))
plt.bar(term_risk["Term Group"], term_risk["Default Rate"], edgecolor="black")
plt.title("Default Rate by Loan Term Group")
plt.ylabel("Default Rate")
plt.xlabel("Term Group")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3. EDA observation 2: SBA guarantee portion and default
# ------------------------------------------------------------

print("\n3. DEFAULT RATE BY SBA GUARANTEE PORTION")

df_model["portion_temp"] = df_model["SBA_Appv"] / df_model["GrAppv"]
df_model["portion_temp"] = df_model["portion_temp"].replace([np.inf, -np.inf], np.nan)

df_model["portion_group_temp"] = pd.cut(
    df_model["portion_temp"],
    bins=[0, 0.5, 0.75, 0.9, 1.0],
    labels=["<=50%", "51-75%", "76-90%", "91-100%"],
    include_lowest=True
)

portion_risk = df_model.groupby("portion_group_temp")["y"].agg(["count", "mean"]).reset_index()
portion_risk.columns = ["SBA Guarantee Portion", "Loan Count", "Default Rate"]
portion_risk["Default Rate"] = portion_risk["Default Rate"].round(4)

print(portion_risk)

# Visualization: default rate by SBA guarantee portion
plt.figure(figsize=(8, 5))
plt.bar(portion_risk["SBA Guarantee Portion"].astype(str), portion_risk["Default Rate"], edgecolor="black")
plt.title("Default Rate by SBA Guarantee Portion")
plt.ylabel("Default Rate")
plt.xlabel("SBA Guarantee Portion")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 4. EDA observation 3: State risk
# ------------------------------------------------------------

print("\n4. TOP STATES BY DEFAULT RATE")

state_risk = (
    df_model.groupby("State")["y"]
    .agg(["count", "mean"])
    .reset_index()
)

state_risk.columns = ["State", "Loan Count", "Default Rate"]

# Avoid tiny states with too few observations
state_risk_filtered = state_risk[state_risk["Loan Count"] >= 100].copy()
state_risk_filtered = state_risk_filtered.sort_values("Default Rate", ascending=False)
state_risk_filtered["Default Rate"] = state_risk_filtered["Default Rate"].round(4)

print(state_risk_filtered.head(15))

# Visualization: top 10 risky states
top10_states = state_risk_filtered.head(10)

plt.figure(figsize=(9, 5))
plt.bar(top10_states["State"], top10_states["Default Rate"], edgecolor="black")
plt.title("Top 10 States by Default Rate")
plt.ylabel("Default Rate")
plt.xlabel("State")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 5. Extra table: business type and urban/rural
# ------------------------------------------------------------

print("\n5. DEFAULT RATE BY BUSINESS TYPE AND URBAN/RURAL")

business_location_table = pd.crosstab(
    df_model["NewExist"],
    df_model["UrbanRural"],
    values=df_model["y"],
    aggfunc="mean"
)

business_location_table = business_location_table.round(4)
print(business_location_table)


# ------------------------------------------------------------
# 6. Clean temporary columns
# ------------------------------------------------------------

df_model = df_model.drop(
    columns=["term_group_temp", "portion_temp", "portion_group_temp"],
    errors="ignore"
)


# ------------------------------------------------------------
# HAI AN NOTES
# ------------------------------------------------------------

# Hai An notes:

# 1. EDA observation:
#    The target variable is imbalanced. Most loans are paid in full, while a smaller portion
#    are charged off/defaulted. This means accuracy alone can be misleading, because a model
#    can look accurate by predicting most loans as paid. The team should compare models using
#    AUC, recall for defaults, and especially validation net profit.

# 2. EDA observation:
#    Loan term appears to be a useful predictor. Different term groups show different default
#    rates, which suggests that risk is not evenly distributed across loan maturity.
#    This supports using Term directly and possibly creating a RealEstate feature where
#    Term >= 240 months.

# 3. EDA observation:
#    SBA guarantee portion may contain useful risk information. The default rate changes across
#    different guarantee-ratio groups, suggesting that the relationship between SBA_Appv and
#    GrAppv can help explain loan risk better than using the raw dollar amounts alone.

# Feature ideas:
#    1. Portion = SBA_Appv / GrAppv. This captures the share of the loan guaranteed by SBA.
#    2. RealEstate = 1 if Term >= 240 months, else 0. Long-term loans may behave differently
#       and this is also used as a real-estate proxy.
#    3. same_state_bank = 1 if State == BankState, else 0. Local lenders may have better
#       information about borrowers.
#    4. NAICS_sector = first two digits of NAICS. This captures broad industry risk.
#    5. unguaranteed_amount = GrAppv - SBA_Appv. This captures the bank's exposure.

# Leakage concern:
#    Do not use MIS_Status, ChgOffDate, ChgOffPrinGr, or BalanceGross as predictors because
#    they are known only after the loan outcome. DisbursementGross should also be handled
#    carefully: in this workflow it is used for profit calculation, not as a default predictor.

# Lending interpretation:
#    The bank should not approve loans only based on whether the predicted class is paid/default.
#    It should estimate P(default), then approve loans only when the expected profit is positive.
#    Features such as Term, SBA guarantee portion, State, and UrbanRural can help separate safer
#    loans from higher-risk loans before building KNN and tree-based models.


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

# 1. SBA guarantee portion
df_model["Portion"] = df_model["SBA_Appv"] / df_model["GrAppv"]
df_model["Portion"] = df_model["Portion"].replace([np.inf, -np.inf], np.nan).fillna(0)

df_model["unguaranteed_ratio"] = 1 - df_model["Portion"]
df_model["unguaranteed_amount"] = df_model["GrAppv"] - df_model["SBA_Appv"]

# 2. Job impact
df_model["jobs_total"] = df_model["CreateJob"] + df_model["RetainedJob"]

df_model["jobs_per_dollar"] = df_model["jobs_total"] / df_model["GrAppv"]
df_model["jobs_per_dollar"] = df_model["jobs_per_dollar"].replace([np.inf, -np.inf], np.nan).fillna(0)

# 3. Same-state lender
df_model["same_state_bank"] = np.where(df_model["State"] == df_model["BankState"], 1, 0)

# 4. Real-estate proxy
df_model["RealEstate"] = np.where(df_model["Term"] >= 240, 1, 0)

# 5. NAICS sector
df_model["NAICS_str"] = (
    df_model["NAICS"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(6)
)

df_model["NAICS_sector"] = df_model["NAICS_str"].str[:2]

# 6. Clean LowDoc and RevLineCr
def clean_yes_no(series):
    cleaned = series.astype(str).str.strip().str.upper()
    cleaned = cleaned.replace({
        "Y": "Yes", "YES": "Yes", "1": "Yes",
        "N": "No", "NO": "No", "0": "No",
        "NAN": "Unknown", "": "Unknown",
    })
    cleaned = np.where(pd.Series(cleaned).isin(["Yes", "No"]), cleaned, "Unknown")
    return cleaned

df_model["LowDoc_clean"] = clean_yes_no(df_model["LowDoc"])
df_model["RevLineCr_clean"] = clean_yes_no(df_model["RevLineCr"])

# 7. Log transforms
for col in ["GrAppv", "SBA_Appv", "NoEmp"]:
    if col in df_model.columns:
        df_model[f"log_{col}"] = np.log1p(df_model[col])

# 8. Simple interaction
df_model["RealEstate_x_Portion"] = df_model["RealEstate"] * df_model["Portion"]


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

    "Portion",
    "unguaranteed_ratio",
    "unguaranteed_amount",
    "jobs_total",
    "jobs_per_dollar",
    "same_state_bank",
    "RealEstate",
    "log_GrAppv",
    "log_SBA_Appv",
    "log_NoEmp",
    "RealEstate_x_Portion",
]


if USE_DISBURSEMENT_AS_PREDICTOR:
    numeric_cols += ["DisbursementGross"]

categorical_cols = [
    "State",
    "BankState",
    "NewExist",
    "UrbanRural",
    "NAICS_sector",
    "LowDoc_clean",
    "RevLineCr_clean",

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
preprocess_hgb = Pipeline(steps=[
    ("preprocess", preprocess_tree),
    ("to_dense", FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else X))
])

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
    knn_results = []

    # Best KNN setting from CV:
    # k = 51, weights = distance, p = 2
    knn_best = KNeighborsClassifier(
        n_neighbors=51,
        weights="distance",
        p=2,
        n_jobs=-1
    )

    knn_pipe, knn_prob, knn_theory, knn_tuned, knn_threshold_table = fit_predict_evaluate(
        "KNN_k51_distance",
        knn_best,
        preprocess_obj=preprocess_scaled
    )

    print("\nKNN validation results:")
    print(pd.DataFrame([knn_theory, knn_tuned]))

    print("\nTop 10 KNN thresholds by validation net profit:")
    print(knn_threshold_table.head(10))

    best_knn_tuned = knn_tuned

# Hai An KNN notes:
# First run result:
# Best CV setting: k=51, weights="distance", p=2
# Best validation threshold:
# Validation net profit:
# Approved default rate:
# Interpretation:
# KNN was evaluated using preprocess_scaled because it depends on distance.
# The final KNN decision should be based on validation net profit, not AUC alone.

# Hai An KNN notes:
# First run result: Tested 8 models (k=11,31,51,101 × uniform/distance)
#                  Best: KNN_k51_distance
# Best validation threshold: 0.218 (if default prob > 0.218 → deny)
# Validation net profit:     48,458,033.80
# Approved default rate:     8.26% (of approved loans, 8.26% actually defaulted)
# Interpretation:
#   - distance weighting consistently beats uniform across all k values
#   - k=51 is the sweet spot: k=11 is too noisy, k=101 over-smooths
#     (recall increases but precision drops → too many good customers denied)
#   - AUC=0.823, Brier=0.108 → probability calibration is reasonably good
#   - Downside: ~29s runtime for k51_distance, slower than tree-based models



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
    tree_results = []

    # ------------------------------------------------------------
    # Candidate 1: Random Forest primary candidate
    # Best CV AUC setting
    # ------------------------------------------------------------

    rf_primary = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=50,
        max_features=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf_primary_pipe, rf_primary_prob, rf_primary_theory, rf_primary_tuned, rf_primary_threshold_table = fit_predict_evaluate(
        "RF_depth12_leaf50_maxfeatNone",
        rf_primary,
        preprocess_obj=preprocess_tree
    )

    tree_results.append(rf_primary_tuned)

    print("\nRandom Forest primary candidate validation results:")
    print(pd.DataFrame([rf_primary_theory, rf_primary_tuned]))

    print("\nTop 10 RF primary thresholds by validation net profit:")
    print(rf_primary_threshold_table.head(10))


    # ------------------------------------------------------------
    # Candidate 2: Random Forest robustness check
    # Feature randomness version
    # ------------------------------------------------------------

    rf_half_features = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=50,
        max_features=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf_half_pipe, rf_half_prob, rf_half_theory, rf_half_tuned, rf_half_threshold_table = fit_predict_evaluate(
        "RF_depth12_leaf50_maxfeat0.5",
        rf_half_features,
        preprocess_obj=preprocess_tree
    )

    tree_results.append(rf_half_tuned)

    print("\nRandom Forest max_features=0.5 candidate validation results:")
    print(pd.DataFrame([rf_half_theory, rf_half_tuned]))

    print("\nTop 10 RF max_features=0.5 thresholds by validation net profit:")
    print(rf_half_threshold_table.head(10))

    # ------------------------------------------------------------
    # Candidate 3: HistGradientBoosting baseline from team lead
    # ------------------------------------------------------------

    hgb_team = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE
    )

    hgb_team_pipe, hgb_team_prob, hgb_team_theory, hgb_team_tuned, hgb_team_threshold_table = fit_predict_evaluate(
        "HGB_iter200_lr0.05_leaf31",
        hgb_team,
        preprocess_obj=preprocess_hgb
    )

    tree_results.append(hgb_team_tuned)

    print("\nHistGradientBoosting team config validation results:")
    print(pd.DataFrame([hgb_team_theory, hgb_team_tuned]))

    print("\nTop 10 HGB team config thresholds by validation net profit:")
    print(hgb_team_threshold_table.head(10))


    # ------------------------------------------------------------
    # Candidate 4: HistGradientBoosting tuning around team config
    # ------------------------------------------------------------

    hgb_grid_results = []

    for max_iter in [150, 200, 250]:
        for learning_rate in [0.03, 0.05, 0.08]:
            for max_leaf_nodes in [15, 31, 63]:
                hgb = HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=RANDOM_STATE
                )

                hgb_pipe, hgb_prob, hgb_theory, hgb_tuned, hgb_threshold_table = fit_predict_evaluate(
                    f"HGB_iter{max_iter}_lr{learning_rate}_leaf{max_leaf_nodes}",
                    hgb,
                    preprocess_obj=preprocess_hgb
                )

                hgb_grid_results.append(hgb_tuned)
                tree_results.append(hgb_tuned)

                print(
                    f"HGB iter={max_iter}, lr={learning_rate}, leaf={max_leaf_nodes}: "
                    f"profit={hgb_tuned['net_profit']:,.2f}, "
                    f"threshold={hgb_tuned['threshold_default']:.3f}"
                )

    hgb_grid_df = pd.DataFrame(hgb_grid_results).sort_values(
        "net_profit",
        ascending=False
    ).reset_index(drop=True)

    print("\nBest HGB tuning results:")
    print(hgb_grid_df.head(10))
    
    # ------------------------------------------------------------
    # Compare tree-family candidates
    # ------------------------------------------------------------

    tree_results_df = pd.DataFrame(tree_results).sort_values(
        "net_profit",
        ascending=False
    ).reset_index(drop=True)

    print("\nTree-family validation comparison:")
    print(tree_results_df)

    best_tree_tuned = tree_results_df.iloc[0].to_dict()

    print("\nBest tree-family model:")
    print(best_tree_tuned)

# Hai An tree-family notes:
# Feature set:
# Basic SBA numeric fields plus engineered Hai An features:
# Portion, unguaranteed_ratio, unguaranteed_amount, jobs_total,
# jobs_per_dollar, same_state_bank, RealEstate, log_GrAppv,
# log_SBA_Appv, log_NoEmp, RealEstate_x_Portion,
# and categorical features State, BankState, NewExist, UrbanRural,
# NAICS_sector, LowDoc_clean, RevLineCr_clean.
#
# Best RF result:
# RF_depth12_leaf50_maxfeatNone
# Validation profit = 66,333,240.30
# Threshold = 0.1687
# AUC = 0.9523
#
# HGB team config:
# HGB_iter200_lr0.05_leaf31
# Validation profit = 67,543,479.55
# Threshold = 0.1092
# AUC = 0.9606
#
# Best HGB tuning result:
# HGB_iter250_lr0.08_leaf63
# Validation profit = 68,323,835.70
# Threshold = 0.0893
# AUC = 0.9609
# Approval rate = 71.46%
# Approved default rate = 1.67%
#
# Interpretation:
# HistGradientBoosting outperformed Random Forest on validation profit using Hai An's feature set.
# The best HGB model uses a stricter default threshold, approving loans only when predicted
# default probability is <= 8.93%. This improves profit by reducing approved defaults while
# still approving about 71% of validation loans. Test set was not used.


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
    
if RUN_KNN_CV:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    knn_cv_results = []

    k_values = [11, 31, 51, 101]
    weights_options = ["uniform", "distance"]

    for k in k_values:
        for w in weights_options:
            candidate = Pipeline(steps=[
                ("preprocess", preprocess_scaled),
                ("model", KNeighborsClassifier(
                    n_neighbors=k,
                    weights=w,
                    p=2,
                    n_jobs=-1
                )),
            ])

            scores = cross_val_score(
                candidate,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1
            )

            result = {
                "owner": "Hai An",
                "model_family": "KNN",
                "params": {"k": k, "weights": w, "p": 2},
                "cv_auc_mean": float(np.mean(scores)),
                "cv_auc_std": float(np.std(scores)),
            }

            knn_cv_results.append(result)
            add_cv_result("Hai An", "KNN", {"k": k, "weights": w, "p": 2}, scores)

            print(f"KNN k={k}, weights={w}: CV AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    knn_cv_df = pd.DataFrame(knn_cv_results).sort_values("cv_auc_mean", ascending=False)
    print("\nBest KNN CV results:")
    print(knn_cv_df)

# Hai An KNN CV notes:
# Best CV setting:      k=51, weights="distance", p=2 → CV AUC = 0.8301 ± 0.0058
# Did CV agree with validation profit?
#   Yes - both methods ranked k=51 distance as #1 and k=11 as worst.
#   distance consistently beats uniform at every k value in both CV and validation.
#   CV is trustworthy for KNN hyperparameter selection on this dataset.


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
    
if RUN_TREE_CV:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    tree_cv_results = []

    for depth in [12, 16]:
        for leaf in [50, 100]:
            for max_feat in [None, 0.5, "sqrt"]:
                candidate = Pipeline(steps=[
                    ("preprocess", preprocess_tree),
                    ("model", RandomForestClassifier(
                        n_estimators=50,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        max_features=max_feat,
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    )),
                ])

                scores = cross_val_score(
                    candidate,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1
                )

                params = {
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "max_features": max_feat,
                    "n_estimators": 50,
                }

                result = {
                    "owner": "Hai An",
                    "model_family": "Random Forest",
                    "params": params,
                    "cv_auc_mean": float(np.mean(scores)),
                    "cv_auc_std": float(np.std(scores)),
                }

                tree_cv_results.append(result)
                add_cv_result("Hai An", "Random Forest", params, scores)

                print(
                    f"RF depth={depth}, leaf={leaf}, max_features={max_feat}: "
                    f"CV AUC = {scores.mean():.4f} (+/- {scores.std():.4f})"
                )

    tree_cv_df = pd.DataFrame(tree_cv_results).sort_values("cv_auc_mean", ascending=False)
    print("\nBest Tree-family CV results:")
    print(tree_cv_df)

# Hai An tree CV notes:
# Best CV setting:    depth=12, leaf=50, max_features=None → CV AUC = 0.9516 ± 0.0016
#                     (depth=16 identical — prefer depth=12 for simplicity)
# Did max_features matter?
#   Yes, hugely - the single most important parameter in this grid.
#   None >> 0.5 >> sqrt, with sqrt dropping ~0.06-0.07 AUC below None.
#   max_depth barely mattered (12 vs 16 gap < 0.0001).
#   min_samples_leaf had a small consistent effect (50 > 100).
#   Conclusion: always use max_features=None for this dataset.


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
# → 3 candidates only:
#   1. RF: depth=12, leaf=50, max_features=None  (top CV AUC = 0.9516, primary candidate)
#   2. RF: depth=12, leaf=50, max_features=0.5   (CV AUC = 0.9468, sanity check on max_features)
#   3. KNN: k=51, distance                        (CV AUC = 0.8301, best KNN representative)
#
# Skip depth=16 (identical to 12), sqrt (too weak), leaf=100 (consistently worse),
# and all weaker KNN variants - already settled from earlier validation run.



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
