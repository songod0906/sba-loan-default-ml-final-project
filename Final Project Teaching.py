# -*- coding: utf-8 -*-
"""
SBA Loan Approval ML Project — Learning / Demonstration File v3

1. Business scoring contract
2. Target creation
3. Leakage control
4. Risk-indicator EDA
5. Feature engineering: RealEstate, Recession, Portion, interactions
6. Train / validation / test split
7. Train-only preprocessing
8. Cross-validation idea
9. Required model families
10. Threshold tuning by net profit
11. Gains / lift / profit curve
12. Final decision interpretation

Run cell by cell with #%%.
The dataset file should be named: SBAnational.csv

Model output: P(default)
Business decision: approve / deny
Primary business metric: validation net profit
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

# Teaching mode:
# The full SBA dataset is large. For learning, we use a stratified working sample.
# This keeps the run fast enough for practice.
# Later, the official workflow should set USE_TEACHING_SAMPLE = False and run on full data.
USE_TEACHING_SAMPLE = True
TEACHING_SAMPLE_N = 50000


#%% 2. Load data

df_raw = pd.read_csv("SBAnational.csv")

print("Rows, columns before teaching sample:", df_raw.shape)
print(df_raw.head())
print(df_raw.columns)

#%% 3. Scoring contract

"""
Official business rule from the competition:

Actual paid in full + grant loan:      +5% * DisbursementGross
Actual default + grant loan:           -5 * 5% * DisbursementGross = -25% * DisbursementGross
Deny loan:                              0

Target coding in this file:
y = 1 means default / charged off / higher risk
y = 0 means paid in full / lower risk

Model output:
P(default)

Decision:
Approve loan if P(default) <= threshold
Deny loan if P(default) > threshold

Theoretical break-even threshold:
Expected profit approve = (1-p)*(0.05D) + p*(-0.25D)
                         = 0.05D - 0.30Dp
Approve if 0.05D - 0.30Dp > 0
p < 1/6 = 0.1667

But this theoretical threshold only works perfectly if probabilities are calibrated.
So later we will also tune the threshold on validation profit.
"""

THEORETICAL_DEFAULT_THRESHOLD = 1 / 6

print("Theoretical default threshold:", THEORETICAL_DEFAULT_THRESHOLD)
print("Theoretical success cutoff:", 1 - THEORETICAL_DEFAULT_THRESHOLD)


#%% 4A. Create target variable

# Always restart from the raw data.
# This prevents mistakes when we rerun cells out of order.

df = df_raw.copy()

print(df["MIS_Status"].value_counts(dropna=False))

# Remove rows where the outcome is unknown.
# Missing MIS_Status should not be treated as paid in full.
df = df[df["MIS_Status"].notna()].copy()

df["y"] = np.where(df["MIS_Status"] == "CHGOFF", 1, 0)

print(df["y"].value_counts())
print("Overall default rate:", df["y"].mean())
print("Rows after removing missing MIS_Status:", df.shape)

#%% 4B. Take teaching sample

if USE_TEACHING_SAMPLE:
    sample_n = min(TEACHING_SAMPLE_N, len(df))

    if sample_n < len(df):
        df_sample, _ = train_test_split(
            df,
            train_size=sample_n,
            random_state=RANDOM_STATE,
            stratify=df["y"]
        )

        df = df_sample.sort_index().copy()

    else:
        df = df.copy()

    print("Using teaching sample:", df.shape)
    print("Sample default rate:", df["y"].mean())

else:
    print("Using full dataset:", df.shape)
    print("Full dataset default rate:", df["y"].mean())


#%% 5. Clean money columns

# Money columns often contain "$" and "," symbols.
# We convert them to float so they can be used in modeling and profit calculation.

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


#%% 6. Leakage audit

"""
Leakage = using information the bank would not know when making the approval decision.

Hard leakage columns:
- MIS_Status: it's our target
- ChgOffDate: only known after the loan defaults
- ChgOffPrinGr: charged-off amount, only known after default
- BalanceGross: outstanding balance after loan performance, not approval-time information

Identifier / text columns are also not useful for this learning file:
- LoanNr_ChkDgt, Name, City, Zip, Bank

DisbursementGross is somewhat a different story.
It does not directly reveal default, and the competition uses it for profit.
For this run, GrAppv and SBA_Appv are cleaner predictors.
So this file uses DisbursementGross for profit calculation, but does not use it as a default predictor unless you turn the option on.
"""

USE_DISBURSEMENT_AS_PREDICTOR = False

leakage_cols = ["MIS_Status", "ChgOffDate", "ChgOffPrinGr", "BalanceGross"]
id_text_cols = ["LoanNr_ChkDgt", "Name", "City", "Zip", "Bank"]

df_model = df.drop(columns=leakage_cols + id_text_cols, errors="ignore").copy()

print("Model dataframe shape:", df_model.shape)

print(df_model.columns)


#%% 7. Feature engineering: business risk features

# 7.1 SBA guarantee ratio / Portion
# Portion = percentage of approved loan guaranteed by SBA.
# Higher guarantee means lower loss exposure for the bank, but it may also be associated with riskier loan structure.
# That's basically SBA's job: to let bank approve risker loans to groups like new businesses or cylically volatile businesses.

df_model["Portion"] = df_model["SBA_Appv"] / df_model["GrAppv"]
df_model["Portion"] = df_model["Portion"].replace([np.inf, -np.inf], np.nan)
df_model["Portion"] = df_model["Portion"].fillna(0)

df_model["unguaranteed_ratio"] = 1 - df_model["Portion"]
df_model["unguaranteed_amount"] = df_model["GrAppv"] - df_model["SBA_Appv"]

# 7.2 Job impact features
# Loans can create values, which we can quantify by jobs created. Higher value loans can indicate that
# the businesses are in a healthy state. 

df_model["jobs_total"] = df_model["CreateJob"] + df_model["RetainedJob"]
df_model["jobs_per_dollar"] = df_model["jobs_total"] / df_model["GrAppv"]
df_model["jobs_per_dollar"] = df_model["jobs_per_dollar"].replace([np.inf, -np.inf], np.nan)
df_model["jobs_per_dollar"] = df_model["jobs_per_dollar"].fillna(0)

# 7.3 Same-state bank
# Banks that are from the same states as the borrowers understand the businesses' behaviors more than banks from different states.

df_model["same_state_bank"] = np.where(df_model["State"] == df_model["BankState"], 1, 0)

# 7.4 RealEstate proxy
# The SBA case paper uses Term >= 240 months as a proxy for loans backed by real estate.
# In history, we have seen that the 2008 financial crisis is due to the unhealthy loans backed by real estate. 

df_model["RealEstate"] = np.where(df_model["Term"] >= 240, 1, 0)

# 7.5 Date features + Recession exposure
# Recession exposure means the loan was active during the Great Recession window.
# We use an overlap rule:
# loan start <= recession end AND loan estimated maturity >= recession start

for col in ["ApprovalDate", "DisbursementDate"]:
    if col in df_model.columns:
        df_model[col] = pd.to_datetime(df_model[col], errors="coerce")

recession_start = pd.Timestamp("2007-12-01")
recession_end = pd.Timestamp("2009-06-30")

# Approximate maturity date = disbursement date + Term months.
# We use 30 days per month for simplicity, matching the teaching case approximation.
df_model["estimated_maturity_date"] = df_model["DisbursementDate"] + pd.to_timedelta(df_model["Term"].fillna(0) * 30, unit="D")

df_model["Recession"] = np.where(
    (df_model["DisbursementDate"] <= recession_end) &
    (df_model["estimated_maturity_date"] >= recession_start),
    1,
    0
)

# 7.6 Calendar features known at approval/disbursement time
# There are some hidden calendar features that can affect the default probability too, so let's account for it. 

df_model["approval_year"] = df_model["ApprovalDate"].dt.year
df_model["disbursement_year"] = df_model["DisbursementDate"].dt.year

# ApprovalFY can be dirty text. Convert to numeric where possible.
if "ApprovalFY" in df_model.columns:
    df_model["ApprovalFY_clean"] = (
        df_model["ApprovalFY"]
        .astype(str)
        .str.replace("A", "", regex=False)
        .str.strip()
    )
    df_model["ApprovalFY_clean"] = pd.to_numeric(df_model["ApprovalFY_clean"], errors="coerce")

# 7.7 NAICS sector
# First 2 digits of NAICS = broad industry sector.
# Different sectors have different risk profiles, of course.

df_model["NAICS_str"] = df_model["NAICS"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
df_model["NAICS_sector"] = df_model["NAICS_str"].str[:2]

# 7.8 Clean dirty categorical variables

print("Raw LowDoc values:")
print(df_model["LowDoc"].value_counts(dropna=False).head(20))

print("Raw RevLineCr values:")
print(df_model["RevLineCr"].value_counts(dropna=False).head(20))

def clean_yes_no(series):
    cleaned = series.astype(str).str.strip().str.upper()
    cleaned = cleaned.replace({
        "Y": "Yes",
        "YES": "Yes",
        "1": "Yes",
        "N": "No",
        "NO": "No",
        "0": "No",
        "NAN": "Unknown",
        "": "Unknown",
    })
    cleaned = np.where(pd.Series(cleaned).isin(["Yes", "No"]), cleaned, "Unknown")
    return cleaned

if "LowDoc" in df_model.columns:
    df_model["LowDoc_clean"] = clean_yes_no(df_model["LowDoc"])

if "RevLineCr" in df_model.columns:
    df_model["RevLineCr_clean"] = clean_yes_no(df_model["RevLineCr"])
    
print("Cleaned LowDoc values:")
print(df_model["LowDoc_clean"].value_counts(dropna=False))

print("Cleaned RevLineCr values:")
print(df_model["RevLineCr_clean"].value_counts(dropna=False))

# 7.9 Log-transform skewed loan amount variables
# log1p(x) = log(1+x), safe when x can be 0.

for col in ["GrAppv", "SBA_Appv", "DisbursementGross", "NoEmp"]:
    if col in df_model.columns:
        df_model[f"log_{col}"] = np.log1p(df_model[col])

# 7.10 Interaction features suggested by the original SBA teaching case
# The original SBA teaching case suggests that the effect of SBA guarantee portion
# may differ for real-estate loans and recession-exposed loans.
# Tree-based models can learn some interaction patterns automatically through splits.
# Linear models such as logistic regression need interaction terms created explicitly.

df_model["RealEstate_x_Portion"] = df_model["RealEstate"] * df_model["Portion"]
df_model["Recession_x_Portion"] = df_model["Recession"] * df_model["Portion"]
df_model["Recession_x_RealEstate"] = df_model["Recession"] * df_model["RealEstate"]

print(df_model[["Portion", "RealEstate", "Recession", "RealEstate_x_Portion", "Recession_x_Portion"]].head())

# Let's check all the new features that we added
feature_check_cols = [
    "Portion",
    "unguaranteed_ratio",
    "unguaranteed_amount",
    "jobs_total",
    "jobs_per_dollar",
    "same_state_bank",
    "RealEstate",
    "NAICS_sector",
    "Recession",
    "approval_year",
    "disbursement_year",
    "ApprovalFY_clean",
    "log_GrAppv",
    "log_SBA_Appv",
    "log_NoEmp",
    "RealEstate_x_Portion",
    "Recession_x_Portion",
    "Recession_x_RealEstate",
]

existing_feature_cols = [col for col in feature_check_cols if col in df_model.columns]
print(df_model[existing_feature_cols].head())
print(df_model[existing_feature_cols].isna().sum())

# NAICS sector 00 can be understood as missing or unclassified industry, so, that unnamed sector can have their own risk profile.

#%% 8. Risk-indicator EDA

# This section exists because the project asks us to identify predictors using descriptive statistics and visualization.
# We need to understand risk patterns before seeing model results.

def default_rate_table(data, group_col, y_col="y", min_count=100):
    temp = data[[group_col, y_col]].copy()
    out = temp.groupby(group_col)[y_col].agg(["count", "mean"]).reset_index()
    out = out.rename(columns={"mean": "default_rate"})
    out = out[out["count"] >= min_count]
    return out.sort_values("default_rate", ascending=False)

# Add y into df_model for EDA only.
df_model["y"] = df["y"]

eda_group_cols = [
    "State", "BankState", "NAICS_sector", "NewExist", "UrbanRural",
    "LowDoc_clean", "RevLineCr_clean", "RealEstate", "Recession"
]

for col in eda_group_cols:
    if col in df_model.columns:
        print("\nDefault rate by", col)
        print(default_rate_table(df_model, col, min_count=100).head(15))

# Numeric summaries by outcome
numeric_risk_cols = [
    "Term", "NoEmp", "CreateJob", "RetainedJob", "GrAppv", "SBA_Appv",
    "DisbursementGross", "Portion", "unguaranteed_ratio", "jobs_total",
    "jobs_per_dollar"
]

numeric_risk_cols = [c for c in numeric_risk_cols if c in df_model.columns]

print("\nNumeric summaries by outcome:")
print(df_model.groupby("y")[numeric_risk_cols].median().T)

# Example plot: default rate by RealEstate
if "RealEstate" in df_model.columns:
    realestate_rate = default_rate_table(df_model, "RealEstate", min_count=1)
    plt.figure(figsize=(6, 4))
    plt.bar(realestate_rate["RealEstate"].astype(str), realestate_rate["default_rate"])
    plt.title("Default Rate by RealEstate Proxy")
    plt.xlabel("RealEstate = 1 if Term >= 240 months")
    plt.ylabel("Default Rate")
    plt.tight_layout()
    plt.show()
    
print("\nOverall sample default rate:", df_model["y"].mean())


#%% 9. Select predictors

# Use DisbursementGross for profit calculation.
# Predictor list uses approval-time style variables by default.

base_numeric_cols = [
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
    "Recession",
    "approval_year",
    "disbursement_year",
    "ApprovalFY_clean",
    "log_GrAppv",
    "log_SBA_Appv",
    "log_NoEmp",
    "RealEstate_x_Portion",
    "Recession_x_Portion",
    "Recession_x_RealEstate",
]

if USE_DISBURSEMENT_AS_PREDICTOR:
    base_numeric_cols += ["DisbursementGross", "log_DisbursementGross"]

categorical_cols = [
    "State",
    "BankState",
    "NAICS_sector",
    "NewExist",
    "UrbanRural",
    "LowDoc_clean",
    "RevLineCr_clean",
]

numeric_cols = [c for c in base_numeric_cols if c in df_model.columns]
categorical_cols = [c for c in categorical_cols if c in df_model.columns]

X = df_model[numeric_cols + categorical_cols].copy() # Use a copy so we don't accidentally contaminate the original data
y = df_model["y"].copy()
amount = df_model["DisbursementGross"].copy()

print("Numeric predictors:", numeric_cols)
print("Categorical predictors:", categorical_cols)
print("X shape before encoding:", X.shape)

# Basic missing checks
print("\nMissing numeric values:")
print(X[numeric_cols].isna().sum().sort_values(ascending=False).head(20))

print("\nMissing categorical values:")
print(X[categorical_cols].isna().sum().sort_values(ascending=False).head(20))

# Convert categorical predictors to string before splitting.
# This prevents OneHotEncoder from seeing mixed types like 1.0 and "Unknown".

for col in categorical_cols:
    X[col] = X[col].fillna("Unknown").astype(str)

print("\nCategorical dtypes after conversion:")
print(X[categorical_cols].dtypes)

#%% 10. Train / validation / test split

# Learning-file split:
# train = model fitting + cross-validation
# validation = threshold tuning + model comparison + gains/lift
# test = final untouched check after the workflow is finalized

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

print("Train shape:", X_train.shape, "Default rate:", y_train.mean())
print("Valid shape:", X_valid.shape, "Default rate:", y_valid.mean())
print("Test shape:", X_test.shape, "Default rate:", y_test.mean())


#%% 11. Train-only preprocessing

# Split numeric and categorical data
X_train_num = X_train[numeric_cols].copy()
X_valid_num = X_valid[numeric_cols].copy()
X_test_num = X_test[numeric_cols].copy()

X_train_cat = X_train[categorical_cols].copy()
X_valid_cat = X_valid[categorical_cols].copy()
X_test_cat = X_test[categorical_cols].copy()

# Fill numeric missing values using TRAIN medians only
train_medians = X_train_num.median()

X_train_num = X_train_num.fillna(train_medians)
X_valid_num = X_valid_num.fillna(train_medians)
X_test_num = X_test_num.fillna(train_medians)

# Fill categorical missing values
X_train_cat = X_train_cat.fillna("Unknown")
X_valid_cat = X_valid_cat.fillna("Unknown")
X_test_cat = X_test_cat.fillna("Unknown")

# One-hot encode categorical variables
X_train_cat_dummy = pd.get_dummies(X_train_cat, drop_first=True)
X_valid_cat_dummy = pd.get_dummies(X_valid_cat, drop_first=True)
X_test_cat_dummy = pd.get_dummies(X_test_cat, drop_first=True)

# Align validation/test columns to training dummy columns
X_valid_cat_dummy = X_valid_cat_dummy.reindex(columns=X_train_cat_dummy.columns, fill_value=0)
X_test_cat_dummy = X_test_cat_dummy.reindex(columns=X_train_cat_dummy.columns, fill_value=0)

# Combine numeric and dummy variables
X_train_ready = pd.concat([X_train_num, X_train_cat_dummy], axis=1)
X_valid_ready = pd.concat([X_valid_num, X_valid_cat_dummy], axis=1)
X_test_ready = pd.concat([X_test_num, X_test_cat_dummy], axis=1)

print("Train ready shape:", X_train_ready.shape)
print("Valid ready shape:", X_valid_ready.shape)
print("Test ready shape:", X_test_ready.shape)

print("Missing values in train:", X_train_ready.isna().sum().sum())
print("Missing values in valid:", X_valid_ready.isna().sum().sum())
print("Missing values in test:", X_test_ready.isna().sum().sum())

# Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_ready)
X_valid_scaled = scaler.transform(X_valid_ready)
X_test_scaled = scaler.transform(X_test_ready)

print("Scaled train shape:", X_train_scaled.shape)
print("Scaled valid shape:", X_valid_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)

print("Mean of first scaled train column:", X_train_scaled[:, 0].mean())
print("Std of first scaled train column:", X_train_scaled[:, 0].std())

# For tree models, scaling is not necessary, but using one common preprocessing pipeline is simpler for this learning file.
# The official workflow can later optimize preprocessing by model family.


# We can also use pipeline preprocessing to speed up the process, we dont have to do it manually as above.
# This lets model blocks use the raw X_train / X_valid tables while still fitting
# imputation, encoding, and scaling inside the training data only.

# Preprocessing for models that need scaling:
# KNN, Logistic, Neural Network, LDA/QDA

numeric_transformer_scaled = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Preprocessing for tree-based models:
# Decision Tree, Bagging, Random Forest, Boosting
# Trees do not need scaling.

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

#%% 12. Business evaluation helper functions

def loan_profit_vector(y_true, decision_approve, amount_series):
    """
    y_true: 1 = default, 0 = paid
    decision_approve: True = approve, False = deny
    amount_series: DisbursementGross
    """
    amount_arr = np.asarray(amount_series, dtype=float)
    y_arr = np.asarray(y_true)
    approve_arr = np.asarray(decision_approve).astype(bool)

    gain_if_paid = 0.05 * amount_arr
    loss_if_default = -0.25 * amount_arr

    profit = np.where(
        approve_arr,
        np.where(y_arr == 0, gain_if_paid, loss_if_default),
        0.0,
    )
    return profit


def evaluate_prob_model(model_name, y_true, prob_default, amount_series, threshold):
    """
    Evaluate one model at one default-probability threshold.
    Approve if P(default) <= threshold.
    """
    y_true_arr = np.asarray(y_true)
    prob_default_arr = np.asarray(prob_default)

    pred_default = (prob_default_arr > threshold).astype(int)
    decision_approve = prob_default_arr <= threshold

    confmat = confusion_matrix(y_true_arr, pred_default, labels=[1, 0])

    TP = confmat[0, 0]  # actual default, predicted default / denied
    FN = confmat[0, 1]  # actual default, predicted paid / approved -> dangerous
    FP = confmat[1, 0]  # actual paid, predicted default / denied -> lost opportunity
    TN = confmat[1, 1]  # actual paid, predicted paid / approved

    profit = loan_profit_vector(y_true_arr, decision_approve, amount_series)

    result = {
        "model": model_name,
        "threshold_default": threshold,
        "threshold_success": 1 - threshold,
        "accuracy": accuracy_score(y_true_arr, pred_default),
        "recall_default_sensitivity": recall_score(y_true_arr, pred_default, zero_division=0),
        "specificity_paid": TN / (TN + FP) if (TN + FP) > 0 else np.nan,
        "precision_default": precision_score(y_true_arr, pred_default, zero_division=0),
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
    return result


def tune_threshold_by_profit(model_name, y_true, prob_default, amount_series, thresholds=None):
    """
    Search thresholds on validation set and choose the one with highest net profit.
    This is the business version of threshold tuning.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 120)

    # Always include the theoretical break-even threshold.
    thresholds = np.unique(
        np.append(thresholds, THEORETICAL_DEFAULT_THRESHOLD)
    )

    rows = []
    for th in thresholds:
        rows.append(
            evaluate_prob_model(
                model_name,
                y_true,
                prob_default,
                amount_series,
                th
            )
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("net_profit", ascending=False).reset_index(drop=True)
    return out


def fit_predict_evaluate(
    model_name,
    estimator,
    threshold=THEORETICAL_DEFAULT_THRESHOLD,
    preprocess_obj=preprocess_scaled
):
    start = time.perf_counter()

    pipe = Pipeline(steps=[
        ("preprocess", preprocess_obj),
        ("model", estimator),
    ])

    pipe.fit(X_train, y_train)
    prob_default_valid = pipe.predict_proba(X_valid)[:, 1]

    runtime = time.perf_counter() - start

    theoretical_result = evaluate_prob_model(
        model_name=model_name,
        y_true=y_valid,
        prob_default=prob_default_valid,
        amount_series=amount_valid,
        threshold=threshold,
    )
    theoretical_result["threshold_type"] = "theoretical_1_over_6"
    theoretical_result["runtime_seconds"] = runtime

    threshold_table = tune_threshold_by_profit(
        model_name=model_name,
        y_true=y_valid,
        prob_default=prob_default_valid,
        amount_series=amount_valid,
    )
    tuned_result = threshold_table.iloc[0].to_dict()
    tuned_result["threshold_type"] = "validation_profit_tuned"
    tuned_result["runtime_seconds"] = runtime

    return pipe, prob_default_valid, theoretical_result, tuned_result, threshold_table

#%% 12A. How to read the helper functions

"""
Why do we use helper functions here?

In normal class labs, we usually write the evaluation code directly after each model:

1. Fit model
2. Predict probability
3. Choose threshold
4. Convert probability into approve / deny decision
5. Build confusion matrix
6. Calculate accuracy, recall, precision, AUC
7. Calculate net profit

That is easy when we only run one model.

But in this project, we need to compare many models:
- KNN
- Decision Tree
- Bagging
- Random Forest
- Boosting
- Ridge / Lasso / ElasticNet Logistic Regression
- Neural Network
- LDA / QDA

If we copy-paste the same evaluation code eight times, we may make mistakes.
So we put the repeated logic inside helper functions.

"""

"""
Function map:

1. loan_profit_vector()
   Input:
   - actual y values
   - approve / deny decision
   - loan amount

   Output:
   - profit or loss for each loan

2. evaluate_prob_model()
   Input:
   - model name
   - actual y values
   - predicted probability of default
   - loan amount
   - threshold

   Output:
   - one row of metrics:
     accuracy, recall, specificity, precision, F1, AUC, Brier score,
     net profit, approval rate, default rate among approved loans

3. tune_threshold_by_profit()
   Input:
   - predicted probabilities
   - actual y values
   - loan amount

   Output:
   - a table of thresholds sorted by net profit

4. fit_predict_evaluate()
   Input:
   - model name
   - sklearn model
   - preprocessing object

   Output:
   - fitted pipeline
   - validation predicted probabilities
   - result at theoretical threshold
   - result at validation-tuned threshold
   - full threshold table
"""

"""
The core business flow is:

predicted P(default)
        ↓
compare with threshold
        ↓
approve if P(default) <= threshold
deny if P(default) > threshold
        ↓
calculate profit:
approved + paid     = +5% * DisbursementGross
approved + default  = -25% * DisbursementGross
denied              = 0
        ↓
compare models by validation net profit
"""
#%% 12B. Tiny manual example of the business rule

# Suppose we have 5 validation loans.
# y = 1 means default, y = 0 means paid.

example_y = np.array([0, 1, 0, 1, 0])

# Suppose the model predicts these default probabilities:
example_prob_default = np.array([0.05, 0.40, 0.12, 0.20, 0.08])

# Suppose these are the loan amounts:
example_amount = np.array([100000, 100000, 200000, 50000, 80000])

# Use the theoretical threshold:
example_threshold = 1 / 6

# Approve if predicted default probability is low enough
example_approve = example_prob_default <= example_threshold

print("Approve decisions:")
print(example_approve)

# Calculate profit loan by loan
example_profit = loan_profit_vector(
    y_true=example_y,
    decision_approve=example_approve,
    amount_series=example_amount
)

print("Profit per loan:")
print(example_profit)

print("Total profit:")
print(example_profit.sum())

# Now use the full evaluation function
example_result = evaluate_prob_model(
    model_name="Tiny example",
    y_true=example_y,
    prob_default=example_prob_default,
    amount_series=example_amount,
    threshold=example_threshold
)

print(pd.DataFrame([example_result]))

#%% 12C. How to understand the pipeline

"""
In earlier class labs, we usually did preprocessing manually:

1. Fill missing values
2. Create dummy variables
3. Scale X_train
4. Scale X_valid
5. Fit model

That is what Block 11 showed manually.

A Pipeline does the same steps, but packages them together.

Why use pipeline here?

Because for every model, we need to repeat preprocessing correctly.
The pipeline makes sure:
- medians are learned from training data only
- dummy variables are learned from training data only
- scaling is learned from training data only
- validation data receives the same transformation

So this:

pipe = Pipeline([
    ("preprocess", preprocess_scaled),
    ("model", knn_estimator)
])

means:

raw X_train
    → clean missing values
    → one-hot encode categories
    → scale numeric variables
    → fit KNN

raw X_valid
    → apply the same preprocessing
    → predict probability of default
"""

"""
We use two preprocessing objects:

preprocess_scaled:
- used for models that need scaling
- KNN, logistic regression, neural network, LDA/QDA

preprocess_tree:
- used for tree models
- decision tree, bagging, random forest, boosting
- no scaling for numeric variables, because trees do not need scaling
- this keeps tree split thresholds easier to interpret
"""

#%% 13. Baseline decisions

# Baseline 1: approve every loan. Let's see the profit if we approve everything
approve_all_profit = loan_profit_vector(
    y_true=y_valid,
    decision_approve=np.ones(len(y_valid), dtype=bool),
    amount_series=amount_valid,
).sum()

# Baseline 2: deny every loan.
deny_all_profit = 0.0

print("Approve-all validation profit:", approve_all_profit)
print("Deny-all validation profit:", deny_all_profit)


#%% 14. Required model family 1 — KNN

"""
KNN = K-Nearest Neighbors.

Idea:
For each validation loan, KNN asks:
"Which training loans look most similar to this loan?"

Then it uses the default behavior of those neighbors to estimate:

P(default)

Why scaling matters:
KNN uses distance. If one variable has a huge scale, it can dominate the distance.
So KNN should use preprocess_scaled.

Important:
fit_predict_evaluate() does the repeated project workflow:
1. fit preprocessing + model on training data
2. predict P(default) on validation data
3. evaluate the theoretical threshold = 1/6
4. tune threshold by validation net profit
5. return both results
"""

RUN_KNN = True

if RUN_KNN:
    # Model settings:
    # n_neighbors = how many similar historical loans to look at.
    # weights="distance" means closer neighbors matter more.
    # p=2 means Euclidean distance.
    knn_estimator = KNeighborsClassifier(
        n_neighbors=51,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )

    # Return values:
    # knn_pipe = fitted preprocessing + KNN model
    # knn_prob = validation predicted probabilities of default
    # knn_theory = result using threshold 1/6
    # knn_tuned = result using best validation-profit threshold
    # knn_threshold_table = all tested thresholds sorted by net profit
    knn_pipe, knn_prob, knn_theory, knn_tuned, knn_threshold_table = fit_predict_evaluate(
        model_name="KNN",
        estimator=knn_estimator,
        preprocess_obj=preprocess_scaled,
    )

    print("KNN summary: theoretical threshold vs validation-profit-tuned threshold")
    print(pd.DataFrame([knn_theory, knn_tuned]))

    print("\nTop KNN thresholds by validation net profit:")
    print(knn_threshold_table.head(10))

else:
    print("Skipping KNN. Set RUN_KNN = True to run.")
    
#%% 15. Required model family 2 — Single classification tree

"""
Decision Tree idea:

A tree predicts risk by asking a sequence of yes/no questions.

Example:
- Is Term <= 59.5?
- Is Recession <= 0.5?
- Is same_state_bank <= 0.5?

Each final leaf contains historical loans.
The default probability is estimated from the default rate inside that leaf.

Why use preprocess_tree:
Trees do not need scaling.
Keeping numeric variables unscaled makes splits easier to interpret.
For example:
Term <= 59.5
is easier to understand than:
scaled Term <= -0.43
"""

single_tree = DecisionTreeClassifier(
    max_depth=6,             # limit tree depth to reduce overfitting
    min_samples_leaf=500,    # each leaf must contain enough loans
    random_state=RANDOM_STATE,
)

tree_pipe, tree_prob, tree_theory, tree_tuned, tree_threshold_table = fit_predict_evaluate(
    model_name="Decision Tree",
    estimator=single_tree,
    preprocess_obj=preprocess_tree,
)

print("Decision Tree summary: theoretical threshold vs validation-profit-tuned threshold")
print(pd.DataFrame([tree_theory, tree_tuned]))

print("\nTop Decision Tree thresholds by validation net profit:")
print(tree_threshold_table.head(10))


#%% 16. Explainable tree visualization

"""

This is a teaching tree.
I intentionally keep it shallow so you can understand how a decision tree works.

The purpose is to show:
1. what kind of questions a tree asks
2. how loans move from root node to leaf node
3. how each leaf estimates paid/default proportions
"""

explain_tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=1000,
    random_state=RANDOM_STATE,
)

explain_pipe = Pipeline(steps=[
    ("preprocess", preprocess_tree),
    ("model", explain_tree),
])

explain_pipe.fit(X_train, y_train)

# Get feature names after preprocessing.
# ColumnTransformer adds prefixes like num__ and cat__.
# We clean them to make the tree easier to read.
feature_names = explain_pipe.named_steps["preprocess"].get_feature_names_out()

feature_names = [
    name.replace("num__", "").replace("cat__", "")
    for name in feature_names
]

plt.figure(figsize=(28, 12))

plot_tree(
    explain_pipe.named_steps["model"],
    feature_names=feature_names,
    class_names=["Paid", "Default"],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=9,
)

plt.title("Small Explanation Tree")
plt.tight_layout()
plt.show()

#%% 17. Required model family 3 — Bagging

"""
Bagging = Bootstrap Aggregating.

Main idea:
A single tree can be unstable.
Small changes in the training data can create a different tree.

Bagging reduces that instability by training many trees.

How it works:
1. Take many bootstrap samples from the training data.
   Bootstrap sample = sample rows with replacement.
2. Fit one tree on each sample.
3. Average their predicted probabilities.

For classification:
Each tree estimates P(default).
Bagging averages the P(default) values across all trees.

Why this can help:
One tree may overreact to noise.
Many trees averaged together are usually more stable.

Why use preprocess_tree:
Bagging is still a tree-based model.
Trees do not need scaling.
"""

# This is the tree used inside the bagging model.
# It can be deeper than the explanation tree because bagging averages many trees.
base_tree = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=100,
    random_state=RANDOM_STATE,
)

# BaggingClassifier trains many versions of base_tree.
bagging = BaggingClassifier(
    estimator=base_tree,
    n_estimators=30,       # number of trees
    max_samples=0.7,       # each tree sees 70% of training rows
    bootstrap=True,        # sample rows with replacement
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

bag_pipe, bag_prob, bag_theory, bag_tuned, bag_threshold_table = fit_predict_evaluate(
    model_name="Bagging",
    estimator=bagging,
    preprocess_obj=preprocess_tree,
)

print("Bagging summary: theoretical threshold vs validation-profit-tuned threshold")
print(pd.DataFrame([bag_theory, bag_tuned]))

print("\nTop Bagging thresholds by validation net profit:")
print(bag_threshold_table.head(10))


#%% 18. Required model family 4 — Random Forest

"""
Random Forest = bagging + random feature selection.

Bagging:
- trains many trees on bootstrapped samples of rows

Random Forest:
- also trains many trees on bootstrapped samples of rows
- but at each split, each tree only considers a subset of features

Why this can help:
If many trees keep choosing the same dominant feature, the trees become too similar.
Random feature selection makes trees more different from each other.
Averaging different trees can improve generalization.

Important project nuance:
In our SBA data, Term is extremely strong.
If max_features is too small, some trees may not see Term at important splits.
That can hurt performance.
So we include max_features=None here as a learning starting point,
meaning all features can compete at each split.

Why use preprocess_tree:
Random Forest is tree-based, so scaling is not needed.
"""

rf = RandomForestClassifier(
    n_estimators=100,       # number of trees
    max_depth=16,           # control tree complexity
    min_samples_leaf=50,    # prevent tiny unstable leaves
    max_features=None,      # allow all features at each split
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

rf_pipe, rf_prob, rf_theory, rf_tuned, rf_threshold_table = fit_predict_evaluate(
    model_name="Random Forest",
    estimator=rf,
    preprocess_obj=preprocess_tree,
)

print("Random Forest summary: theoretical threshold vs validation-profit-tuned threshold")
print(pd.DataFrame([rf_theory, rf_tuned]))

print("\nTop Random Forest thresholds by validation net profit:")
print(rf_threshold_table.head(10))

# Random Forest feature importance
# This tells us which variables were most useful for splitting across the forest.
rf_model = rf_pipe.named_steps["model"]
rf_feature_names = rf_pipe.named_steps["preprocess"].get_feature_names_out()

# Clean feature names
rf_feature_names = [
    name.replace("num__", "").replace("cat__", "")
    for name in rf_feature_names
]

rf_importance = pd.DataFrame({
    "feature": rf_feature_names,
    "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop Random Forest feature importances:")
print(rf_importance.head(25))

plt.figure(figsize=(10, 8))
plt.barh(
    rf_importance.head(20)["feature"][::-1],
    rf_importance.head(20)["importance"][::-1]
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


#%% 19. Required model family 5 — Boosting

"""
Boosting idea:

Bagging / Random Forest:
- train many trees independently
- average them

Boosting:
- train trees sequentially
- each new tree pays more attention to observations the previous trees handled poorly

AdaBoost:
- starts with equal weight on all training observations
- after each weak tree, misclassified observations receive more weight
- the next tree tries harder on those difficult cases

Why weak trees:
AdaBoost usually uses shallow trees.
Each tree is weak alone, but the sequence of trees can become strong.

Why use preprocess_tree:
This boosting model is built from decision trees.
Trees do not need scaled numeric variables.
"""

boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=2,               # weak learner: shallow tree
        min_samples_leaf=200,      # avoid tiny unstable leaves
        random_state=RANDOM_STATE,
    ),
    n_estimators=100,              # number of weak trees added sequentially
    learning_rate=0.05,            # smaller = slower, more cautious learning
    random_state=RANDOM_STATE,
)

boost_pipe, boost_prob, boost_theory, boost_tuned, boost_threshold_table = fit_predict_evaluate(
    model_name="AdaBoost",
    estimator=boosting,
    preprocess_obj=preprocess_tree,
)

print("AdaBoost summary: theoretical threshold vs validation-profit-tuned threshold")
print(pd.DataFrame([boost_theory, boost_tuned]))

print("\nTop AdaBoost thresholds by validation net profit:")
print(boost_threshold_table.head(10))

#%% 20. Required model family 6 — Logistic regression: Ridge, Lasso, ElasticNet

"""
Solver:
- SAGA supports L1 and ElasticNet in sklearn.
- We use SAGA for consistency across Ridge, Lasso, and ElasticNet.
- If n_iter reaches max_iter, the model may not have fully converged.
"""

def print_logit_diagnostics(model_name, pipe, top_n=20):
    """
    Print convergence info and largest coefficients for a fitted logistic pipeline.

    """
    model = pipe.named_steps["model"]

    print(f"\n{model_name} n_iter:")
    print(model.n_iter_)

    if np.max(model.n_iter_) >= model.max_iter:
        print("Warning: model reached max_iter. Consider increasing max_iter.")

    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names
    ]

    coef = model.coef_[0]

    coef_table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False)

    print(f"\nTop {top_n} {model_name} coefficients by absolute value:")
    print(coef_table.head(top_n))

    print(f"\nNumber of nonzero coefficients in {model_name}:")
    print(np.sum(coef != 0))

    return coef_table


# -------------------------
# Ridge Logistic Regression
# -------------------------

logit_ridge = LogisticRegression(
    penalty="l2",
    C=1.0,                  # larger C = weaker regularization
    solver="saga",
    max_iter=3000,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

ridge_pipe, ridge_prob, ridge_theory, ridge_tuned, ridge_threshold_table = fit_predict_evaluate(
    model_name="Logistic Ridge",
    estimator=logit_ridge,
    preprocess_obj=preprocess_scaled,
)

print("Ridge Logistic summary: theoretical threshold vs validation-profit-tuned threshold")
print(pd.DataFrame([ridge_theory, ridge_tuned]))

print("\nTop Ridge Logistic thresholds by validation net profit:")
print(ridge_threshold_table.head(10))

ridge_coef_table = print_logit_diagnostics("Ridge Logistic", ridge_pipe)


# -------------------------
# Lasso Logistic Regression
# -------------------------

logit_lasso = LogisticRegression(
    penalty="l1",
    C=0.5,                  # smaller C = stronger regularization
    solver="saga",
    max_iter=3000,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

lasso_pipe, lasso_prob, lasso_theory, lasso_tuned, lasso_threshold_table = fit_predict_evaluate(
    model_name="Logistic Lasso",
    estimator=logit_lasso,
    preprocess_obj=preprocess_scaled,
)

print("Lasso Logistic summary: theoretical threshold vs validation-profit-tuned threshold")
print(pd.DataFrame([lasso_theory, lasso_tuned]))

print("\nTop Lasso Logistic thresholds by validation net profit:")
print(lasso_threshold_table.head(10))

lasso_coef_table = print_logit_diagnostics("Lasso Logistic", lasso_pipe)


# -----------------------------
# ElasticNet Logistic Regression
# -----------------------------

logit_elastic = LogisticRegression(
    penalty="elasticnet",
    C=0.5,
    l1_ratio=0.5,           # 0 = Ridge, 1 = Lasso, 0.5 = halfway
    solver="saga",
    max_iter=3000,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

elastic_pipe, elastic_prob, elastic_theory, elastic_tuned, elastic_threshold_table = fit_predict_evaluate(
    model_name="Logistic ElasticNet",
    estimator=logit_elastic,
    preprocess_obj=preprocess_scaled,
)

print("ElasticNet Logistic summary: theoretical threshold vs validation-profit-tuned threshold")
print(pd.DataFrame([elastic_theory, elastic_tuned]))

print("\nTop ElasticNet Logistic thresholds by validation net profit:")
print(elastic_threshold_table.head(10))

elastic_coef_table = print_logit_diagnostics("ElasticNet Logistic", elastic_pipe)

#%% 21. Required model family 7 — Neural Network

"""
Neural Network idea:

A neural network learns nonlinear patterns by combining many weighted inputs.

For this project:
- Input = cleaned loan features
- Output = P(default)

Why use preprocess_scaled:
Neural networks are optimization-based.
If variables are on very different scales, training becomes harder.
So neural networks should use preprocess_scaled.

Architecture:
hidden_layer_sizes=(64, 32)

This means:
- first hidden layer has 64 neurons
- second hidden layer has 32 neurons

Regularization:
alpha = L2 penalty.
Higher alpha means stronger regularization.

Early stopping:
early_stopping=True means sklearn keeps 10% of the training data internally
to monitor whether the neural network is still improving.
This is separate from our validation set.
Our validation set is still used for threshold tuning and model comparison.
"""

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    alpha=0.0001,
    early_stopping=True,
    validation_fraction=0.10,
    max_iter=100,
    random_state=RANDOM_STATE,
)

mlp_pipe, mlp_prob, mlp_theory, mlp_tuned, mlp_threshold_table = fit_predict_evaluate(
    model_name="Neural Network MLP",
    estimator=mlp,
    preprocess_obj=preprocess_scaled,
)

print("Neural Network summary: theoretical threshold vs validation-profit-tuned threshold")
print(pd.DataFrame([mlp_theory, mlp_tuned]))

print("\nTop Neural Network thresholds by validation net profit:")
print(mlp_threshold_table.head(10))

mlp_model = mlp_pipe.named_steps["model"]

print("\nMLP n_iter:")
print(mlp_model.n_iter_)

print("\nMLP final loss:")
print(mlp_model.loss_)

print("\nMLP best validation score during early stopping:")
print(mlp_model.best_validation_score_)

# Plot training loss curve
plt.figure(figsize=(7, 4))
plt.plot(mlp_model.loss_curve_)
plt.title("Neural Network Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

#%% 22. Required model family 8 — Discriminant analysis

"""
Discriminant Analysis idea:

LDA and QDA estimate the distribution of predictors separately for each class.

For this project:
- class 0 = Paid
- class 1 = Default

Then the model uses Bayes-style logic:

P(default | loan profile)

LDA:
- assumes both classes share the same covariance structure
- simpler and usually more stable

QDA:
- allows each class to have its own covariance structure
- more flexible, but less stable with many features

Why use preprocess_scaled:
LDA/QDA rely on covariance/distance-like structure.
Scaling helps prevent large-scale variables from dominating.

Why convert to dense arrays:
After one-hot encoding, sklearn preprocessing may return a sparse matrix.
LDA/QDA usually work more safely with dense arrays.

Warning:
QDA can be unstable with many dummy variables.
That is why we test regularized QDA using reg_param.
"""

RUN_LDA_QDA = True

if RUN_LDA_QDA:

    # Use scaled preprocessing for discriminant analysis
    X_train_processed = preprocess_scaled.fit_transform(X_train, y_train)
    X_valid_processed = preprocess_scaled.transform(X_valid)

    # Convert to dense if needed
    X_train_da_full = (
        X_train_processed.toarray()
        if hasattr(X_train_processed, "toarray")
        else X_train_processed
    )

    X_valid_da = (
        X_valid_processed.toarray()
        if hasattr(X_valid_processed, "toarray")
        else X_valid_processed
    )

    # Use a sample if needed for memory/speed.
    # In the teaching sample, X_train has only 30,000 rows, so this will usually use all rows.
    np.random.seed(RANDOM_STATE)
    da_train_size = min(30000, X_train_da_full.shape[0])
    train_pos = np.random.choice(
        X_train_da_full.shape[0],
        size=da_train_size,
        replace=False
    )

    X_train_da = X_train_da_full[train_pos]
    y_train_da = y_train.iloc[train_pos]

    # -------------------------
    # LDA
    # -------------------------

    lda_start = time.perf_counter()

    lda = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto"
    )

    lda.fit(X_train_da, y_train_da)
    lda_prob = lda.predict_proba(X_valid_da)[:, 1]

    lda_runtime = time.perf_counter() - lda_start

    lda_theory = evaluate_prob_model(
        model_name="LDA",
        y_true=y_valid,
        prob_default=lda_prob,
        amount_series=amount_valid,
        threshold=THEORETICAL_DEFAULT_THRESHOLD,
    )
    lda_theory["threshold_type"] = "theoretical_1_over_6"
    lda_theory["runtime_seconds"] = lda_runtime

    lda_threshold_table = tune_threshold_by_profit(
        model_name="LDA",
        y_true=y_valid,
        prob_default=lda_prob,
        amount_series=amount_valid,
    )

    lda_tuned = lda_threshold_table.iloc[0].to_dict()
    lda_tuned["threshold_type"] = "validation_profit_tuned"
    lda_tuned["runtime_seconds"] = lda_runtime

    print("LDA summary: theoretical threshold vs validation-profit-tuned threshold")
    print(pd.DataFrame([lda_theory, lda_tuned]))

    print("\nTop LDA thresholds by validation net profit:")
    print(lda_threshold_table.head(10))


    # -------------------------
    # QDA with regularization
    # -------------------------

    qda_results = []
    qda_models = {}

    for reg in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:

        qda_start = time.perf_counter()

        qda = QuadraticDiscriminantAnalysis(
            reg_param=reg
        )

        qda.fit(X_train_da, y_train_da)
        qda_prob = qda.predict_proba(X_valid_da)[:, 1]

        qda_runtime = time.perf_counter() - qda_start

        qda_threshold_table = tune_threshold_by_profit(
            model_name=f"QDA reg={reg}",
            y_true=y_valid,
            prob_default=qda_prob,
            amount_series=amount_valid,
        )

        qda_best = qda_threshold_table.iloc[0].to_dict()
        qda_best["reg_param"] = reg
        qda_best["runtime_seconds"] = qda_runtime

        qda_results.append(qda_best)
        qda_models[reg] = qda

    qda_results_df = pd.DataFrame(qda_results).sort_values(
        "net_profit",
        ascending=False
    )

    print("\nQDA results by best validation net profit:")
    print(qda_results_df)

else:
    print("Skipping discriminant analysis.")

#%% 23. Simple cross-validation example on training data

"""
The assignment asks for hyperparameter selection using cross-validation.

This block demonstrates the idea with Random Forest.

Important separation:

1. Cross-validation happens inside the training data only.
   We use it to compare hyperparameter settings.

2. The validation set is NOT used during cross-validation.
   We use the validation set later for:
   - threshold tuning
   - model comparison
   - validation net profit leaderboard

3. The test set stays untouched until the very end.

Why Random Forest here:
Random Forest is currently our strongest model family, so it is a good example
for showing how cross-validation can help choose hyperparameters.

Why use preprocess_tree:
Random Forest is tree-based. Trees do not need scaled numeric variables.
"""

RUN_CV_DEMO = True # Change = True to run, but its gonna take a lot of time

if RUN_CV_DEMO:

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_rows = []

    # Small grid for teaching/demo.
    # We keep it small because cross-validation trains many models.
    for depth in [12, 16]:
        for leaf in [50, 100]:
            for max_feat in [None, "sqrt", 0.5]:

                candidate = Pipeline(steps=[
                    ("preprocess", preprocess_tree),
                    ("model", RandomForestClassifier(
                        n_estimators=50,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        max_features=max_feat,

                        # Important:
                        # cross_val_score already parallelizes across folds.
                        # Keep the model n_jobs=1 to avoid nested parallelism.
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
                    n_jobs=-1,
                )

                cv_rows.append({
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "max_features": max_feat,
                    "cv_auc_mean": scores.mean(),
                    "cv_auc_std": scores.std(),
                })

                print(
                    "depth:", depth,
                    "| leaf:", leaf,
                    "| max_features:", max_feat,
                    "| mean CV AUC:", scores.mean(),
                )

    cv_results_df = pd.DataFrame(cv_rows).sort_values(
        "cv_auc_mean",
        ascending=False
    )

    print("\nRandom Forest CV results:")
    print(cv_results_df)

else:
    print("Skipping CV demo by default. Set RUN_CV_DEMO = True to run.") 

#%% 24. Build model leaderboard

"""
This block collects all model results into one leaderboard.

Important:
- We report both theoretical threshold and validation-profit-tuned threshold.
- The final model should be selected mainly by validation net profit.
- AUC and Brier score are useful supporting metrics, but profit is the business objective here.
"""

leaderboard_rows = []

# -------------------------
# Baseline rows
# -------------------------

baseline_rows = [
    {
        "model": "Approve All Baseline",
        "threshold_type": "baseline",
        "threshold_default": np.nan,
        "threshold_success": np.nan,
        "auc": np.nan,
        "brier": np.nan,
        "accuracy": (y_valid == 0).mean(),
        "recall_default_sensitivity": 0.0,
        "specificity_paid": 1.0,
        "approval_rate": 1.0,
        "approved_default_rate": y_valid.mean(),
        "denied_default_rate": np.nan,
        "net_profit": approve_all_profit,
        "runtime_seconds": 0.0,
    },
    {
        "model": "Deny All Baseline",
        "threshold_type": "baseline",
        "threshold_default": np.nan,
        "threshold_success": np.nan,
        "auc": np.nan,
        "brier": np.nan,
        "accuracy": (y_valid == 1).mean(),
        "recall_default_sensitivity": 1.0,
        "specificity_paid": 0.0,
        "approval_rate": 0.0,
        "approved_default_rate": np.nan,
        "denied_default_rate": y_valid.mean(),
        "net_profit": deny_all_profit,
        "runtime_seconds": 0.0,
    },
]

leaderboard_rows.extend(baseline_rows)


# -------------------------
# Model result rows
# -------------------------

possible_results = [
    "knn_theory", "knn_tuned",
    "tree_theory", "tree_tuned",
    "bag_theory", "bag_tuned",
    "rf_theory", "rf_tuned",
    "boost_theory", "boost_tuned",
    "ridge_theory", "ridge_tuned",
    "lasso_theory", "lasso_tuned",
    "elastic_theory", "elastic_tuned",
    "mlp_theory", "mlp_tuned",
    "lda_theory", "lda_tuned",
]

for name in possible_results:
    if name in globals():
        row = globals()[name].copy()
        leaderboard_rows.append(row)


# -------------------------
# Add best QDA result
# -------------------------

if "qda_results_df" in globals():
    qda_best = qda_results_df.iloc[0].to_dict()
    qda_best["threshold_type"] = "validation_profit_tuned"
    leaderboard_rows.append(qda_best)


# -------------------------
# Build leaderboard table
# -------------------------

leaderboard = pd.DataFrame(leaderboard_rows)

leaderboard = leaderboard.sort_values(
    "net_profit",
    ascending=False
).reset_index(drop=True)

display_cols = [
    "model",
    "threshold_type",
    "threshold_default",
    "threshold_success",
    "auc",
    "brier",
    "accuracy",
    "recall_default_sensitivity",
    "specificity_paid",
    "approval_rate",
    "approved_default_rate",
    "denied_default_rate",
    "net_profit",
    "runtime_seconds",
]

# Keep only columns that exist, so baseline/QDA rows do not break the display
display_cols = [col for col in display_cols if col in leaderboard.columns]

print("Full validation leaderboard:")
print(leaderboard[display_cols])

leaderboard.to_csv("sba_learning_model_leaderboard.csv", index=False)


# -------------------------
# Optional: tuned-only leaderboard
# -------------------------

tuned_leaderboard = leaderboard[
    leaderboard["threshold_type"].isin([
        "validation_profit_tuned",
        "baseline",
    ])
].copy()

tuned_leaderboard = tuned_leaderboard.sort_values(
    "net_profit",
    ascending=False
).reset_index(drop=True)

print("\nTuned/baseline leaderboard only:")
print(tuned_leaderboard[display_cols])

tuned_leaderboard.to_csv("sba_learning_tuned_leaderboard.csv", index=False)

#%% 25. Gains / lift / profit curve for chosen model

"""

The leaderboard tells us:
Which model + threshold gave the highest validation net profit?

The profit curve tells us:
If we rank validation loans from safest to riskiest, how far down the list
should we approve loans before cumulative profit stops improving?


"""

# -------------------------
# 1. Choose best model
# -------------------------

# Use tuned_leaderboard if it exists.
# This table keeps only baseline rows and validation-profit-tuned model rows.

if "tuned_leaderboard" in globals():
    selection_table = tuned_leaderboard.copy()
else:
    selection_table = leaderboard.copy()

# Remove approve-all / deny-all baselines from model selection.
model_selection_table = selection_table[
    ~selection_table["model"].isin([
        "Approve All Baseline",
        "Deny All Baseline",
    ])
].copy()

# Pick model with highest validation net profit.
best_row = model_selection_table.sort_values(
    "net_profit",
    ascending=False
).iloc[0]

best_model_name = best_row["model"]

print("Best validation model selected for profit curve:")
print(best_row)

# -------------------------
# 2. Map model name to validation predicted probabilities
# -------------------------

# Each model block created a validation probability vector.
# These probabilities are P(default) for each validation loan.

prob_map = {}

for model_name, prob_name in [
    ("KNN", "knn_prob"),
    ("Decision Tree", "tree_prob"),
    ("Bagging", "bag_prob"),
    ("Random Forest", "rf_prob"),
    ("AdaBoost", "boost_prob"),
    ("Logistic Ridge", "ridge_prob"),
    ("Logistic Lasso", "lasso_prob"),
    ("Logistic ElasticNet", "elastic_prob"),
    ("Neural Network MLP", "mlp_prob"),
    ("LDA", "lda_prob"),
]:
    if prob_name in globals():
        prob_map[model_name] = globals()[prob_name]

if best_model_name not in prob_map:
    raise ValueError(
        f"No probability vector found for {best_model_name}. "
        "Check prob_map or the model block output variable names."
    )

best_prob_default = prob_map[best_model_name]
best_prob_success = 1 - best_prob_default

# -------------------------
# 3. Build validation profit curve table
# -------------------------

validation_profit_df = pd.DataFrame({
    "prob_default": best_prob_default,
    "prob_success": best_prob_success,
    "actual_default": y_valid.values,
    "DisbursementGross": amount_valid.values,
})

# Rank safest loans first.
# Highest P(success) = lowest P(default).
validation_profit_df = validation_profit_df.sort_values(
    "prob_success",
    ascending=False
).reset_index(drop=True)

# Profit if each loan is approved.
# Paid loan:     +5% * DisbursementGross
# Default loan:  -25% * DisbursementGross
validation_profit_df["loan_profit_if_approved"] = loan_profit_vector(
    y_true=validation_profit_df["actual_default"],
    decision_approve=np.ones(len(validation_profit_df), dtype=bool),
    amount_series=validation_profit_df["DisbursementGross"],
)

# Cumulative profit if we approve loans from safest to riskiest.
validation_profit_df["cum_profit"] = validation_profit_df[
    "loan_profit_if_approved"
].cumsum()

validation_profit_df["approval_depth"] = (
    np.arange(len(validation_profit_df)) + 1
) / len(validation_profit_df)

validation_profit_df["approved_count"] = np.arange(len(validation_profit_df)) + 1
validation_profit_df["denied_count"] = len(validation_profit_df) - validation_profit_df["approved_count"]

validation_profit_df["cum_good_loans"] = (
    validation_profit_df["actual_default"] == 0
).cumsum()

validation_profit_df["cum_bad_loans"] = (
    validation_profit_df["actual_default"] == 1
).cumsum()

validation_profit_df["cum_approved_default_rate"] = (
    validation_profit_df["cum_bad_loans"] /
    validation_profit_df["approved_count"]
)

# -------------------------
# 4. Find max-profit approval depth
# -------------------------

max_profit_idx = validation_profit_df["cum_profit"].idxmax()
max_profit_row = validation_profit_df.loc[max_profit_idx]

print("\nBest model for profit curve:", best_model_name)
print("Maximum cumulative validation profit:", max_profit_row["cum_profit"])
print("Approval depth at max profit:", max_profit_row["approval_depth"])
print("Number approved at max profit:", int(max_profit_row["approved_count"]))
print("Number denied at max profit:", int(max_profit_row["denied_count"]))
print("Probability of success cutoff:", max_profit_row["prob_success"])
print("Probability of default cutoff:", max_profit_row["prob_default"])
print("Approved default rate at max profit:", max_profit_row["cum_approved_default_rate"])

print("\nLeaderboard-tuned threshold for this model:")
print("Default threshold:", best_row["threshold_default"])
print("Success threshold:", best_row["threshold_success"])
print("Validation net profit:", best_row["net_profit"])

print("\nProfit curve improvement over approve-all:")
print(max_profit_row["cum_profit"] - approve_all_profit)

print("\nProfit curve improvement over leaderboard-tuned RF:")
print(max_profit_row["cum_profit"] - best_row["net_profit"])

# -------------------------
# 5. Plot cumulative profit curve
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    validation_profit_df["approval_depth"],
    validation_profit_df["cum_profit"],
)

plt.axvline(
    max_profit_row["approval_depth"],
    linestyle="--",
)

plt.title(f"Validation Cumulative Profit Curve — {best_model_name}")
plt.xlabel("Approval Depth: Safest Loans Approved First")
plt.ylabel("Cumulative Net Profit")
plt.tight_layout()
plt.show()

# -------------------------
# 6. Plot cumulative good/default loans
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    validation_profit_df["approval_depth"],
    validation_profit_df["cum_good_loans"],
    label="Cumulative Paid Loans",
)

plt.plot(
    validation_profit_df["approval_depth"],
    validation_profit_df["cum_bad_loans"],
    label="Cumulative Default Loans",
)

plt.axvline(
    max_profit_row["approval_depth"],
    linestyle="--",
)

plt.title(f"Validation Gains Curve — {best_model_name}")
plt.xlabel("Approval Depth: Safest Loans Approved First")
plt.ylabel("Cumulative Count")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 7. Save profit curve table
# -------------------------

validation_profit_df.to_csv(
    "sba_validation_profit_curve.csv",
    index=False,
)


#%% 26. ROC curve and calibration check for chosen model

"""
This block checks the quality of the chosen model's predicted probabilities.

ROC curve:
- Shows ranking ability.
- If AUC is high, the model is good at ranking defaulted loans above paid loans.

Calibration:
- Checks whether predicted probabilities match actual default rates.
- A model can rank loans well but still give probabilities that are too high or too low.

For lending:
- Ranking is useful for approving safest loans first.
- Calibration matters because we use probability cutoffs for business decisions.
"""

# Use the best model selected in Block 25.
# best_prob_default = validation P(default) from the chosen model.
# best_model_name = chosen model name, usually Random Forest in our current run.

chosen_auc = roc_auc_score(y_valid, best_prob_default)
chosen_brier = brier_score_loss(y_valid, best_prob_default)

print("Chosen model:", best_model_name)
print("Validation AUC:", chosen_auc)
print("Validation Brier score:", chosen_brier)

# -------------------------
# ROC curve
# -------------------------

fpr, tpr, roc_thresholds = roc_curve(y_valid, best_prob_default)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"{best_model_name} AUC={chosen_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall Default")
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# Calibration by decile
# -------------------------

calibration_df = pd.DataFrame({
    "prob_default": best_prob_default,
    "actual_default": y_valid.values,
})

# Sort loans into 10 groups by predicted default probability.
# duplicates="drop" prevents errors if many loans have the same predicted probability.
calibration_df["decile"] = pd.qcut(
    calibration_df["prob_default"],
    q=10,
    duplicates="drop",
)

calibration_table = calibration_df.groupby("decile", observed=False).agg(
    avg_pred_default=("prob_default", "mean"),
    actual_default_rate=("actual_default", "mean"),
    count=("actual_default", "count"),
).reset_index()

print("\nCalibration table:")
print(calibration_table)

plt.figure(figsize=(6, 5))
plt.plot(
    calibration_table["avg_pred_default"],
    calibration_table["actual_default_rate"],
    marker="o",
)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title(f"Calibration Check — {best_model_name}")
plt.xlabel("Average Predicted Default Probability")
plt.ylabel("Actual Default Rate")
plt.tight_layout()
plt.show()

calibration_table.to_csv("sba_calibration_table.csv", index=False)


#%% 27. Error analysis at chosen threshold

"""
This block studies where the chosen model makes mistakes.

Most dangerous mistake:
- approved loan that actually defaulted
- this creates a large loss

Opportunity-cost mistake:
- denied loan that would have paid in full
- this loses potential profit, but does not create direct loss

Correct decisions:
- approve paid loan
- deny default loan
"""

# Choose one final validation threshold.
# We prefer the profit-curve cutoff if Block 25 created max_profit_row.
# Otherwise, use the leaderboard-tuned threshold.

if "max_profit_row" in globals():
    final_default_threshold = float(max_profit_row["prob_default"])
else:
    final_default_threshold = float(best_row["threshold_default"])

final_success_threshold = 1 - final_default_threshold

print("Final validation policy threshold:")
print("Approve if P(default) <=", final_default_threshold)
print("Equivalent: approve if P(success) >=", final_success_threshold)

# Apply policy to validation loans.
chosen_approve = best_prob_default <= final_default_threshold
chosen_pred_default = (best_prob_default > final_default_threshold).astype(int)

valid_error_df = X_valid.copy()
valid_error_df["actual_default"] = y_valid.values
valid_error_df["prob_default"] = best_prob_default
valid_error_df["prob_success"] = 1 - best_prob_default
valid_error_df["approved"] = chosen_approve
valid_error_df["pred_default"] = chosen_pred_default
valid_error_df["DisbursementGross"] = amount_valid.values

valid_error_df["profit"] = loan_profit_vector(
    y_true=y_valid,
    decision_approve=chosen_approve,
    amount_series=amount_valid,
)

valid_error_df["error_type"] = np.select(
    [
        (valid_error_df["actual_default"] == 1) & (valid_error_df["approved"] == True),
        (valid_error_df["actual_default"] == 0) & (valid_error_df["approved"] == False),
        (valid_error_df["actual_default"] == 1) & (valid_error_df["approved"] == False),
        (valid_error_df["actual_default"] == 0) & (valid_error_df["approved"] == True),
    ],
    [
        "Dangerous: approved default",
        "Opportunity cost: denied paid loan",
        "Correct: denied default",
        "Correct: approved paid loan",
    ],
    default="Unknown",
)

# -------------------------
# Error summary
# -------------------------

error_summary = valid_error_df.groupby("error_type").agg(
    count=("error_type", "count"),
    avg_prob_default=("prob_default", "mean"),
    avg_amount=("DisbursementGross", "mean"),
    total_profit=("profit", "sum"),
).sort_values("total_profit")

error_summary["share_of_validation"] = error_summary["count"] / len(valid_error_df)

print("\nError summary:")
print(error_summary)

print("\nError counts:")
print(valid_error_df["error_type"].value_counts())

print("\nProfit by error type:")
print(valid_error_df.groupby("error_type")["profit"].sum().sort_values())


# -------------------------
# Segment error analysis
# -------------------------

segment_cols = [
    "NAICS_sector",
    "State",
    "BankState",
    "RealEstate",
    "Recession",
    "LowDoc_clean",
    "RevLineCr_clean",
]

for col in segment_cols:
    if col in valid_error_df.columns:
        print("\nError breakdown by", col)

        breakdown = pd.crosstab(
            valid_error_df[col],
            valid_error_df["error_type"],
        )

        # Show largest groups first for easier reading.
        breakdown["total"] = breakdown.sum(axis=1)
        breakdown = breakdown.sort_values("total", ascending=False)

        print(breakdown.head(20))


# -------------------------
# Save detailed validation error file
# -------------------------

valid_error_df.to_csv(
    "sba_validation_error_analysis.csv",
    index=False,
)

#%% 27B. Segment error rates for better interpretation

def segment_error_rates(df, segment_col):
    out = df.groupby(segment_col).agg(
        total=("error_type", "count"),
        dangerous_approved_defaults=(
            "error_type",
            lambda s: (s == "Dangerous: approved default").sum()
        ),
        opportunity_cost_denied_paid=(
            "error_type",
            lambda s: (s == "Opportunity cost: denied paid loan").sum()
        ),
        avg_prob_default=("prob_default", "mean"),
        actual_default_rate=("actual_default", "mean"),
        avg_amount=("DisbursementGross", "mean"),
        total_profit=("profit", "sum"),
    ).reset_index()

    out["dangerous_error_rate"] = (
        out["dangerous_approved_defaults"] / out["total"]
    )

    out["opportunity_cost_rate"] = (
        out["opportunity_cost_denied_paid"] / out["total"]
    )

    # Avoid over-reading tiny groups
    out = out[out["total"] >= 100].copy()

    return out.sort_values("dangerous_error_rate", ascending=False)


for col in ["NAICS_sector", "State", "BankState", "RealEstate", "Recession", "LowDoc_clean", "RevLineCr_clean"]:
    if col in valid_error_df.columns:
        print("\nSegment error rates:", col)
        print(segment_error_rates(valid_error_df, col).head(15))
        
#%% 28. Final untouched test evaluation

"""
Only run this after we freeze:

1. predictor list
2. feature engineering
3. preprocessing choice
4. model family
5. hyperparameters
6. threshold selection rule

Important:
The test set is NOT for tuning.
Do not change the model after seeing the test result.

Frozen validation decision from this learning run:
- Model: Random Forest
- Hyperparameters:
    n_estimators=100
    max_depth=16
    min_samples_leaf=50
    max_features=None
- Threshold:
    approve if P(default) <= final_default_threshold
"""

RUN_FINAL_TEST_EVAL = False # Set to True to run

if RUN_FINAL_TEST_EVAL:

    # -------------------------
    # 1. Freeze final threshold
    # -------------------------

    # Prefer profit-curve cutoff if it exists.
    # Otherwise use leaderboard-tuned threshold.
    if "final_default_threshold" in globals():
        frozen_default_threshold = final_default_threshold
    elif "max_profit_row" in globals():
        frozen_default_threshold = float(max_profit_row["prob_default"])
    else:
        frozen_default_threshold = float(best_row["threshold_default"])

    frozen_success_threshold = 1 - frozen_default_threshold

    print("Frozen final policy:")
    print("Approve if P(default) <=", frozen_default_threshold)
    print("Approve if P(success) >=", frozen_success_threshold)


    # -------------------------
    # 2. Refit final model on train + validation
    # -------------------------

    final_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=50,
        max_features=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    final_pipe = Pipeline(steps=[
        ("preprocess", preprocess_tree),
        ("model", final_rf),
    ])

    final_start = time.perf_counter()

    final_pipe.fit(X_train_valid, y_train_valid)

    test_prob_default = final_pipe.predict_proba(X_test)[:, 1]

    final_runtime = time.perf_counter() - final_start


    # -------------------------
    # 3. Evaluate once on test set
    # -------------------------

    test_result = evaluate_prob_model(
        model_name="Final Random Forest Test",
        y_true=y_test,
        prob_default=test_prob_default,
        amount_series=amount_test,
        threshold=frozen_default_threshold,
    )

    test_result["threshold_type"] = "frozen_from_validation"
    test_result["runtime_seconds"] = final_runtime

    print("\nFinal untouched test result:")
    print(pd.DataFrame([test_result]))


    # -------------------------
    # 4. Test baselines for comparison
    # -------------------------

    test_approve_all_profit = loan_profit_vector(
        y_true=y_test,
        decision_approve=np.ones(len(y_test), dtype=bool),
        amount_series=amount_test,
    ).sum()

    test_deny_all_profit = 0.0

    print("\nTest baselines:")
    print("Approve-all test profit:", test_approve_all_profit)
    print("Deny-all test profit:", test_deny_all_profit)
    print("Final model improvement over approve-all:", test_result["net_profit"] - test_approve_all_profit)


    # -------------------------
    # 5. Save test result
    # -------------------------

    test_result_df = pd.DataFrame([test_result])
    test_result_df.to_csv("sba_final_test_result.csv", index=False)

else:
    print("Skipping final test evaluation.")
    print("Set RUN_FINAL_TEST_EVAL = True only after the workflow is frozen.")


#%% 29. Save summary outputs

summary = {
    "approve_all_profit": approve_all_profit,
    "deny_all_profit": deny_all_profit,

    "best_model_name": best_model_name,

    "best_validation_net_profit_from_leaderboard": float(best_row["net_profit"]),
    "best_default_threshold_from_leaderboard": float(best_row["threshold_default"]),
    "best_success_cutoff_from_leaderboard": float(best_row["threshold_success"]),

    "max_profit_validation_profit_from_rank_curve": float(max_profit_row["cum_profit"]),
    "max_profit_approval_depth": float(max_profit_row["approval_depth"]),
    "max_profit_approved_count": int(max_profit_row["approved_count"]),
    "max_profit_denied_count": int(max_profit_row["denied_count"]),
    "max_profit_success_cutoff_from_rank_curve": float(max_profit_row["prob_success"]),
    "max_profit_default_cutoff_from_rank_curve": float(max_profit_row["prob_default"]),
    "max_profit_approved_default_rate": float(max_profit_row["cum_approved_default_rate"]),
}

if "test_result" in globals():
    summary["final_test_net_profit"] = float(test_result["net_profit"])
    summary["final_test_auc"] = float(test_result["auc"])
    summary["final_test_brier"] = float(test_result["brier"])
    summary["final_test_approval_rate"] = float(test_result["approval_rate"])
    summary["final_test_approved_default_rate"] = float(test_result["approved_default_rate"])

summary_df = pd.DataFrame([summary])
print(summary_df)

summary_df.to_csv("sba_learning_summary.csv", index=False)

print("Done. Saved:")
print("- sba_learning_model_leaderboard.csv")
print("- sba_learning_tuned_leaderboard.csv")
print("- sba_validation_profit_curve.csv")
print("- sba_calibration_table.csv")
print("- sba_validation_error_analysis.csv")
print("- sba_learning_summary.csv")

if "test_result" in globals():
    print("- sba_final_test_result.csv")