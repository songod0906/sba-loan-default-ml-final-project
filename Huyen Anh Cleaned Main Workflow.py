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
OWNER_NAME = "Huyen Anh"


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
# USE_DISBURSEMENT_AS_PREDICTOR = False
#
# DisbursementGross is the actual amount disbursed AFTER the loan is approved.
# At the time the bank is deciding whether to approve the loan, this amount
# is not yet known, only the approved amount (GrAppv) is known.
# Therefore, using DisbursementGross as a predictor would be leakage.
# We keep it only for profit calculation (5% gain or 25% loss per loan).
#
# Columns removed:
# - MIS_Status: the target itself: obvious leakage
# - ChgOffDate: only exists if the loan already defaulted: obvious leakage
# - ChgOffPrinGr: the charged-off amount: only known after default - leakage
# - BalanceGross: outstanding balance after loan activity: not known at approval time
# - LoanNr_ChkDgt, Name, City, Zip, Bank: identifiers with no predictive value
#
# Remaining 19 columns look appropriate:
# - State, BankState: geographic info known at application
# - NAICS: industry code known at application
# - Term, NoEmp, GrAppv, SBA_Appv: loan terms known at approval
# - NewExist, FranchiseCode, UrbanRural, RevLineCr, LowDoc: business info at application
# - ApprovalDate, ApprovalFY, DisbursementDate: timing info


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

# Idea 1: Does RevLineCr (revolving credit) predict default?
huyen_anh_revline = df_model.groupby("RevLineCr")["y"].agg(["count", "mean"])
print(huyen_anh_revline)

# Idea 2: Is loan term nonlinearly related to default?
df_model.groupby(pd.cut(df_model["Term"], bins=10))["y"].mean().plot(kind="bar")
plt.title("Default rate by Term bucket")
plt.tight_layout()
plt.show()

# Idea 3: Does LowDoc interact with loan size?
huyen_anh_lowdoc = df_model.groupby("LowDoc")["y"].agg(["count", "mean"])
print(huyen_anh_lowdoc)

# Huyen Anh notes:
# 1. EDA observation:
#    Revolving line of credit (Y) has a much higher default rate (~25%)
#    compared to non-revolving loans (N) at ~15%. The dirty values
#    (0, 1, 2, R, T, `) suggest this column needs cleaning before modeling.

# 2. EDA observation (from Default rate by Term bucket chart):
#    Very short-term loans (0–39 months) have the highest default rate (~50%),
#    and medium-short loans (39–78 months) also default at ~36%.
#    Default rates drop sharply for mid-range terms (78–155 months, ~4–6%),
#    then spike again around 195–233 months (~26%) before dropping again.
#    This U-shaped pattern suggests Term has a nonlinear relationship with default —
#    something a neural network can capture better than a linear model.

# 3. EDA observation (from LowDoc):
#    LowDoc = R (rare category) has a 75% default rate — extremely high.
#    LowDoc = Y (yes, low-documentation loan) has only ~9% default rate,
#    lower than N (~19%). This is counterintuitive and worth flagging —
#    low-doc loans may attract more established borrowers who need less paperwork.
    
# Feature ideas:
# 1. Clean RevLineCr into Yes/No/Unknown (map Y->Yes, N->No, everything else->Unknown)
#    and use it as a categorical predictor — the gap between Y and N is meaningful.
# 2. Create a RealEstate flag: Term >= 240 months likely = real estate loan.
#    Real estate loans tend to be lower risk (collateral backed), which the
#    chart supports (low default rates at 233–311 month range).
    
# Leakage concern:
#    DisbursementDate is post-approval timing, the exact date money was released
#    may not be known at decision time and could introduce subtle leakage
#    if used as a raw predictor. Safe to extract only the year or a recession flag instead.
    
# Lending interpretation:
#    Banks should be especially cautious with very short-term loans and
#    revolving credit lines — both show disproportionately high default rates.
#    A neural network is well-suited here because the nonlinear Term pattern
#    and interaction between LowDoc and RevLineCr are hard to capture linearly.


#%% 7. Shared feature engineering workspace

# 7.1 SBA guarantee portion
df_model["Portion"] = df_model["SBA_Appv"] / df_model["GrAppv"]
df_model["Portion"] = df_model["Portion"].replace([np.inf, -np.inf], np.nan).fillna(0)
df_model["unguaranteed_ratio"] = 1 - df_model["Portion"]
df_model["unguaranteed_amount"] = df_model["GrAppv"] - df_model["SBA_Appv"]

# 7.2 Job impact 
df_model["jobs_total"] = df_model["CreateJob"] + df_model["RetainedJob"]
df_model["jobs_per_dollar"] = df_model["jobs_total"] / df_model["GrAppv"]
df_model["jobs_per_dollar"] = df_model["jobs_per_dollar"].replace([np.inf, -np.inf], np.nan).fillna(0)

# 7.3 Same-state lender
df_model["same_state_bank"] = np.where(df_model["State"] == df_model["BankState"], 1, 0)

# 7.4 Real-estate proxy (EDA Block 6C feature idea 2)
df_model["RealEstate"] = np.where(df_model["Term"] >= 240, 1, 0)

# 7.5 Recession flag
for col in ["ApprovalDate", "DisbursementDate"]:
    df_model[col] = pd.to_datetime(df_model[col], errors="coerce")

recession_start = pd.Timestamp("2007-12-01")
recession_end = pd.Timestamp("2009-06-30")
df_model["estimated_maturity_date"] = df_model["DisbursementDate"] + pd.to_timedelta(df_model["Term"].fillna(0) * 30, unit="D")
df_model["Recession"] = np.where(
    (df_model["DisbursementDate"] <= recession_end) &
    (df_model["estimated_maturity_date"] >= recession_start),
    1,
    0,
)
df_model["approval_year"] = df_model["ApprovalDate"].dt.year
df_model["disbursement_year"] = df_model["DisbursementDate"].dt.year

# Clean ApprovalFY
df_model["ApprovalFY_clean"] = (
    df_model["ApprovalFY"]
    .astype(str)
    .str.replace("A", "", regex=False)
    .str.strip()
)
df_model["ApprovalFY_clean"] = pd.to_numeric(df_model["ApprovalFY_clean"], errors="coerce")

# 7.6 NAICS sector
df_model["NAICS_str"] = df_model["NAICS"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
df_model["NAICS_sector"] = df_model["NAICS_str"].str[:2]

# 7.7 Clean LowDoc and RevLineCr (EDA Block 6C feature idea 1)
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

# 7.8 Log transforms
for col in ["GrAppv", "SBA_Appv", "NoEmp"]:
    if col in df_model.columns:
        df_model[f"log_{col}"] = np.log1p(df_model[col])

# 7.9 Interaction terms
df_model["RealEstate_x_Portion"] = df_model["RealEstate"] * df_model["Portion"]
df_model["Recession_x_Portion"] = df_model["Recession"] * df_model["Portion"]
df_model["Recession_x_RealEstate"] = df_model["Recession"] * df_model["RealEstate"]

# Quick check after you create features:
print("Current df_model columns:")
print(df_model.columns)


#%% 8. Register predictors

numeric_cols = [
    "Term", "NoEmp", "CreateJob", "RetainedJob", "GrAppv", "SBA_Appv",
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
    numeric_cols += ["DisbursementGross"]

categorical_cols = [
    "State",
    "BankState",
    "NewExist",
    "UrbanRural",
    # Engineered categorical features from Block 7
    "NAICS_sector",
    "LowDoc_clean",
    "RevLineCr_clean",
]

# Keep only columns that actually exist.
numeric_cols = [c for c in numeric_cols if c in df_model.columns]
categorical_cols = [c for c in categorical_cols if c in df_model.columns]

print("Numeric predictors:", numeric_cols)
print("Categorical predictors:", categorical_cols)

# Answer / notes:
# Why did you choose these predictors?
#
# All predictors are known to the bank at loan application time — no leakage risk.
#
# NUMERIC PREDICTORS:
#
# Core loan terms (known at application):
# - Term: EDA (Block 6C) confirmed nonlinear relationship with default —
#   very short (0-39 months, ~50%) and medium-short (39-78 months, ~36%)
#   have the highest default rates. Neural networks can capture this nonlinearity.
# - NoEmp: business size proxy, larger firms tend to be more financially stable
# - CreateJob, RetainedJob: signals business growth potential and economic impact
# - GrAppv: total loan amount approved by bank
# - SBA_Appv: SBA guaranteed portion of the loan
#
# SBA guarantee features:
# - Portion: SBA_Appv / GrAppv — the guarantee ratio. Higher SBA coverage
#   may indicate riskier loan structure (SBA's job is to help riskier borrowers)
# - unguaranteed_ratio: 1 - Portion — the bank's unprotected exposure share
# - unguaranteed_amount: GrAppv - SBA_Appv — the actual dollar amount the
#   bank loses if the loan defaults without SBA coverage
#
# Job impact features:
# - jobs_total: CreateJob + RetainedJob — total economic impact of the loan,
#   higher job impact may signal healthier business activity
# - jobs_per_dollar: jobs_total / GrAppv — job creation efficiency,
#   captures value generated per dollar lent
#
# Geographic/lender feature:
# - same_state_bank: 1 if borrower and bank are in the same state —
#   local banks have better knowledge of borrower's local business environment
#
# Business risk proxies:
# - RealEstate: Term >= 240 months proxy — real-estate backed loans are
#   lower risk due to collateral, confirmed by EDA Term chart (Block 6C idea 2)
# - Recession: whether loan was active during 2007-2009 recession —
#   economic downturns strongly increase default probability
#
# Time/calendar features:
# - approval_year: year the loan was approved — captures macroeconomic
#   conditions and lending environment at time of approval
# - disbursement_year: year money was actually disbursed — may differ
#   from approval year and captures when business actually started using funds
# - ApprovalFY_clean: cleaned fiscal year of SBA commitment —
#   provides additional time granularity beyond calendar year
#
# Log transforms (reduce right skew in large monetary/count variables):
# - log_GrAppv, log_SBA_Appv: loan amount variables
#   are heavily right-skewed. Log transforms help the neural network learn
#   more efficiently by compressing the range of large values
# - log_NoEmp: employee count is also right-skewed, log transform stabilizes it
#
# Interaction terms (explicitly capture combined effects):
# - RealEstate_x_Portion: real-estate loans with low SBA coverage may behave
#   differently — the bank bears more risk on already-collateralized loans
# - Recession_x_Portion: during recession, level of SBA backing matters more —
#   higher guarantee may have been protective during the financial crisis
# - Recession_x_RealEstate: real-estate loans during recession are especially
#   relevant given the 2008 housing crisis was driven by real-estate defaults
#   Note: even though neural networks can learn interactions automatically,
#   explicit terms help the model identify these known domain-relevant patterns faster
#
# CATEGORICAL PREDICTORS:
#
# - State: geographic risk differences across US states
# - BankState: lender location, may reflect different lending standards
#   and risk appetite across states
# - NewExist: new vs existing business — new businesses (code 2) carry
#   higher default risk due to lack of operating history
# - UrbanRural: location type affects business survival and market access —
#   rural businesses may face different economic conditions
# - NAICS_sector: industry sector (first 2 digits of NAICS code) —
#   some industries default more than others (e.g. retail vs manufacturing)
# - LowDoc_clean: cleaned LowDoc program flag — EDA showed meaningful
#   default rate differences across categories (Block 6C observation 3).
#   Cleaned from dirty raw values (A, C, R, S) into Yes/No/Unknown
# - RevLineCr_clean: cleaned revolving credit flag — EDA showed Y has ~25%
#   default rate vs N at ~15%, strong predictive signal (Block 6C idea 1).
#   Cleaned from dirty raw values (0, 1, 2, T, R) into Yes/No/Unknown


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

    # Step 1: Build best NN from CV results
    mlp_best = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),  # CV winner
        activation="relu",
        solver="adam",
        alpha=0.01,                         # CV winner
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )

    mlp_pipe, mlp_prob, mlp_theory, mlp_tuned, mlp_threshold_table = fit_predict_evaluate(
        "Neural Network", mlp_best, preprocess_obj=preprocess_scaled
    )

    # Step 2: Print results at both thresholds
    print("\n--- Results at theoretical threshold (1/6 ≈ 0.167) ---")
    print(pd.DataFrame([mlp_theory])[
        ["model", "threshold_default", "accuracy", "recall_default",
         "specificity_paid", "auc", "net_profit", "approval_rate"]
    ])

    print("\n--- Results at profit-tuned threshold ---")
    print(pd.DataFrame([mlp_tuned])[
        ["model", "threshold_default", "accuracy", "recall_default",
         "specificity_paid", "auc", "net_profit", "approval_rate"]
    ])

    # Step 3: Top 10 thresholds by net profit
    print("\n--- Top 10 thresholds by net profit ---")
    print(mlp_threshold_table.head(10)[
        ["threshold_default", "net_profit", "approval_rate",
         "approved_default_rate", "recall_default", "auc"]
    ])

    # Step 4: Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(mlp_pipe.named_steps["model"].loss_curve_, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Neural Network Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 5: ROC curve
    fpr, tpr, _ = roc_curve(y_valid, mlp_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"NN AUC = {roc_auc_score(y_valid, mlp_prob):.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Neural Network")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# --- Training diagnostics ---
    mlp_model = mlp_pipe.named_steps["model"]
    print("\n--- Training diagnostics ---")
    print("n_iter (epochs run):", mlp_model.n_iter_)
    print("Final training loss:", round(mlp_model.loss_, 4))
    print("Best validation score (early stopping):", round(mlp_model.best_validation_score_, 4))

    # --- Brier score ---
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_valid, mlp_prob)
    print("\n--- Brier Score ---")
    print(f"Brier score: {brier:.4f}")
    print("(lower is better; 0 = perfect, 0.25 = random)")

    # --- Calibration check ---
    calibration_df = pd.DataFrame({
        "prob_default": mlp_prob,
        "actual_default": y_valid.values,
    })
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

    print("\n--- Calibration Table ---")
    print(calibration_table)

    plt.figure(figsize=(6, 5))
    plt.plot(
        calibration_table["avg_pred_default"],
        calibration_table["actual_default_rate"],
        marker="o", label="NN calibration"
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Average Predicted P(default)")
    plt.ylabel("Actual Default Rate")
    plt.title("Calibration Check — Neural Network")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Huyen Anh notes:
#
# === FINAL MODEL ===
# Architecture:     (128, 64, 32) — three hidden layers
# Activation:       ReLU (avoids vanishing gradient vs sigmoid/tanh)
# Solver:           Adam (adaptive learning rates, fast convergence)
# Alpha:            0.01 (L2 regularization)
# Learning rate:    0.001 (Adam default)
# Early stopping:   True (validation_fraction=0.10, n_iter_no_change=10)
# n_iter:           28 epochs
# Final loss:       0.0975
# Best val score:   0.915 (internal early stopping score)
#
# === ARCHITECTURE COMPARISON: Small vs Large ===
# Small (32,16)  alpha=0.05 lr=0.001  → CV AUC 0.9170, profit $57,794,762
# Large (128,64,32) alpha=0.05 lr=0.0005 → CV AUC 0.9152, profit $58,070,693
# Original (128,64,32) alpha=0.01 lr=0.001 → CV AUC 0.9132, profit $59,043,622 ← BEST
#
# Conclusion: original settings beat both new tuning attempts on validation profit.
# Larger model (128,64,32) is justified — it consistently outperforms small (32,16)
# on validation profit across all tuning rounds, despite small model winning CV AUC.
# Always evaluate on validation profit, not CV AUC alone.
#
# === ALPHA SENSITIVITY ===
# alpha=0.01 outperforms alpha=0.05 on validation profit ($59M vs $58M)
# even though alpha=0.05 had higher CV AUC.
# alpha=0.01 provides enough regularization without over-penalizing weights.
#
# === LEARNING RATE EFFECT ===
# lr=0.001 (Adam default) works best for (128,64,32).
# Loss curve declines smoothly from 0.40 to 0.10 over 28 epochs —
# no oscillation or divergence, confirming good convergence.
# Smaller lr=0.0005 did not improve validation profit.
#
# === LOSS CURVE OBSERVATION ===
# Smooth monotonic decline from ~0.40 to ~0.10 over 28 epochs.
# Steep drop in first 2 epochs, then gradual steady decline.
# early_stopping triggered at epoch 28 — model converged cleanly.
# No signs of overfitting (no loss plateau followed by increase).
#
# === ROC CURVE ===
# AUC = 0.910 — strong discrimination between default and paid loans.
# Curve hugs top-left corner especially at low false positive rates,
# meaning the model reliably catches defaults without denying good loans.
#
# === BRIER SCORE / CALIBRATION ===
# Brier score: 0.0771 (well below 0.25 random baseline — good calibration)
# Calibration table shows predicted P(default) closely tracks actual default rate:
#   Decile 1 (P~0.00): actual 0.7%   ← very safe loans correctly identified
#   Decile 5 (P~0.006): actual 3.3%  ← low risk correctly identified
#   Decile 8 (P~0.09): actual 17.6%  ← near population average, well calibrated
#   Decile 9 (P~0.46): actual 44.3%  ← high risk correctly identified
#   Decile 10 (P~0.93): actual 85.7% ← very risky loans correctly identified
# Calibration plot shows points closely following the diagonal (perfect calibration line),
# with slight overestimation in the lowest deciles (predicts near 0 but actual is 0.7-3%).
# Overall calibration is strong — probabilities are reliable for business decisions.
#
# === THRESHOLD SELECTION ===
# Theoretical threshold: 1/6 = 0.1667 → profit $56,964,422
# Profit-tuned threshold: 0.0794 → profit $59,043,622 ← chosen
# Tuned threshold is lower than theoretical, meaning the model is more
# conservative — only approving loans where P(default) <= 7.94%.
# This strict screening results in approved default rate of only 4.51%,
# well below the population default rate of ~17.56%.
#
# === FINAL VALIDATION RESULTS ===
# Best threshold:        P(default) <= 0.0794
# Net profit:            $59,043,622.40
# Approval rate:         74.51%
# Approved default rate: 4.51%
# AUC:                   0.910
# Brier score:           0.0771
# n_iter:                28 epochs
# Final loss:            0.0975

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

RUN_NN_CV = False

if RUN_NN_CV:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Leader's suggestions:
    # 1. Small vs current best architecture
    # 2. Alpha around 0.01 (weaker/stronger)
    # 3. Different learning rates
    architectures = [
        (32, 16),           # small model
        (64, 32),           # medium model
        (128, 64, 32),      # current best
    ]
    alphas = [0.005, 0.01, 0.05]        # around current best 0.01
    learning_rates = [0.0005, 0.001]    # smaller than default

    for arch in architectures:
        for alpha in alphas:
            for lr in learning_rates:
                candidate = Pipeline(steps=[
                    ("preprocess", preprocess_scaled),
                    ("model", MLPClassifier(
                        hidden_layer_sizes=arch,
                        activation="relu",
                        solver="adam",
                        alpha=alpha,
                        learning_rate_init=lr,
                        early_stopping=True,
                        validation_fraction=0.10,
                        n_iter_no_change=10,
                        max_iter=300,
                        random_state=RANDOM_STATE,
                    )),
                ])
                scores = cross_val_score(
                    candidate, X_train, y_train,
                    cv=cv, scoring="roc_auc", n_jobs=1
                )
                add_cv_result(
                    "Huyen Anh", "Neural Network",
                    {"hidden_layer_sizes": arch, "alpha": alpha, "lr": lr},
                    scores
                )
                print(arch, alpha, lr, round(scores.mean(), 4), "±", round(scores.std(), 4))

# Huyen Anh CV notes (Round 2: leader feedback tuning):
# Grid: architectures=(32,16), (64,32), (128,64,32)
#       alphas=0.005, 0.01, 0.05
#       learning_rates=0.0005, 0.001
#
# Key results:
#   Small  (32,16)     alpha=0.05  lr=0.001  → AUC 0.9170 ± 0.0019 (highest CV AUC)
#   Large  (128,64,32) alpha=0.05  lr=0.0005 → AUC 0.9152 ± 0.0019
#   Medium (64,32)     alpha=0.05  lr=0.001  → AUC 0.9117 ± 0.0012 (weakest)
#
# Alpha sensitivity finding:
#   alpha=0.05 consistently outperformed alpha=0.005 and alpha=0.01 in CV AUC.
#   Stronger regularization helps across all architectures on 50k sample.
#
# Learning rate finding:
#   Small model: lr=0.001 slightly better than lr=0.0005
#   Large model: lr=0.0005 slightly better than lr=0.001
#   Difference is small (<0.001 AUC) — not a decisive factor.
#
# Validation profit comparison (overrides CV AUC for final decision):
#   Small  (32,16)     alpha=0.05  lr=0.001  → $57,794,762
#   Large  (128,64,32) alpha=0.05  lr=0.0005 → $58,070,693
#   Original (128,64,32) alpha=0.01 lr=0.001 → $59,043,622 ← WINNER
#
# Final decision: original (128,64,32) alpha=0.01 lr=0.001 chosen.
# CV AUC alone was misleading — small model won CV but lost validation profit.
# Key lesson: always validate hyperparameters on business metric (net profit),
# not just CV AUC. Larger model justified by $1.2M profit advantage over small model.

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
# Two rounds of CV were conducted:
#
# Round 1 (Block 15D first run) — architecture + alpha grid:
#   Grid: (32,), (64,), (64,32), (128,64), (128,64,32), (256,128), (256,128,64), (256,128,64,32)
#         alphas = 0.0001, 0.001, 0.01
#   Best: (128,64,32) alpha=0.01, lr=0.001 → AUC 0.9132 ± 0.0011
#   Finding: 3-layer networks outperform 2-layer and 1-layer.
#            alpha=0.01 consistently better than 0.0001 and 0.001.
#            (256,128,64,32) ties AUC but higher std — not worth extra runtime.
#
# Round 2 (Block 15D second run) — leader feedback tuning:
#   Grid: (32,16), (64,32), (128,64,32)
#         alphas = 0.005, 0.01, 0.05
#         learning_rates = 0.0005, 0.001
#   Best CV AUC: (32,16) alpha=0.05, lr=0.001 → AUC 0.9170 ± 0.0019
#   Finding: stronger alpha=0.05 beats alpha=0.01 in CV AUC across all architectures.
#            small model surprisingly wins CV AUC over large model.
#
# Validation profit comparison (final decision):
#   Small  (32,16)       alpha=0.05 lr=0.001  → $57,794,762
#   Large  (128,64,32)   alpha=0.05 lr=0.0005 → $58,070,693
#   Original (128,64,32) alpha=0.01 lr=0.001  → $59,043,622 ← WINNER
#
# Settings worth taking to validation:
#   ONLY (128,64,32) alpha=0.01 lr=0.001 — highest net profit by $1M+ margin.
#
# Settings NOT worth pursuing:
#   - (32,16): won CV AUC but lost validation profit by $1.2M —
#     CV AUC overstated small model performance
#   - alpha=0.05: better in CV but worse in validation profit
#   - (256,128,64,32): tied CV AUC with (128,64,32) but higher std and slower
#   - (64,32) and smaller: consistently lower AUC and profit than (128,64,32)
#
# Key lesson:
#   CV AUC and validation profit do not always agree.
#   For this competition, validation net profit is the correct selection criterion.
#   The original (128,64,32) alpha=0.01 lr=0.001 was confirmed as final model
#   with $59,043,622 net profit, AUC=0.910, Brier=0.0771.

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
