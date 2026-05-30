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


#%% 2. Load data

df_raw = pd.read_csv("/Users/chuhongminh/Downloads/SBAnational.csv")

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
#DisbursementDate and DisbursementGross
#1. Leakage columns removed because they contain information after loan approval or default event.
#2. ID/text columns removed because they are unique identifiers or free text, causing overfitting.
#3. DisbursementGross is intentionally NOT used as a predictor because:
#  It is the actual loan amount disbursed, known at approval time,
#      but highly correlated with GrAppv and SBA_Appv.
#  Using it may introduce look-ahead bias if not careful.
#  It is already used in profit calculation (amount) separately.
#  Keeping it as predictor could leak future information about loan size vs default risk indirectly.
#  Decision: Keep only GrAppv and SBA_Appv as predictors for loan size.
#
# 4. Remaining columns after cleaning (based on your actual data):
#    - State, BankState, NAICS, ApprovalDate, ApprovalFY, Term, NoEmp,
#      NewExist, CreateJob, RetainedJob, FranchiseCode, UrbanRural,
#      RevLineCr, LowDoc, DisbursementDate, DisbursementGross, GrAppv,
#      SBA_Appv, y

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
# 1. EDA observation: Term differs clearly between default and non-default
term_summary = df_model.groupby("y")["Term"].describe()
print("Term summary by default status:")
print(term_summary)

# 2. EDA observation: NewExist might be a distinguishing factor
new_exist_summary = df_model.groupby("NewExist")["y"].agg(["count", "mean"]).sort_values("mean", ascending=False)
print("\nNewExist summary:")
print(new_exist_summary)

# 3. EDA observation: UrbanRural might have a mild effect
urban_rural_summary = df_model.groupby("UrbanRural")["y"].agg(["count", "mean"]).sort_values("mean", ascending=False)
print("\nUrbanRural summary:")
print(urban_rural_summary)

# Focus idea: variables that may help logistic regression, LDA, or QDA.
# Example only:
# hai_anh_term_summary = df_model.groupby("y")["Term"].median()
# print(hai_anh_term_summary)

# Hai Anh notes:
# 1. EDA observation: EDA observation: Term differs strongly between default and non‑default
#    Defaulted loans (y=1) have much shorter terms (mean ~58 months) than paid‑off loans (mean ~122 months).
#    This suggests shorter‑term loans may be riskier, possibly because they are extended to weaker borrowers.
# 2. EDA observation: NewExist = 2 (likely "Existing business") has highest default rate (~18.9%).
#    NewExist = 1 has ~17.1% default rate.
#    NewExist = 0 (missing / unknown) has very low default rate (~5.3%), but sample size is tiny.
# 3. EDA observation:UrbanRural = 1 (Urban area) has highest default rate (~24.3%).
#    UrbanRural = 2 (Rural) has ~19.4% default rate.
#    UrbanRural = 0 (Unknown) has lowest default rate (~7.0%), but large sample size suggests missing data is systematically different.
# Feature ideas:
# - Use log(Term) or scaled Term for logistic regression / LDA / QDA
# - Create dummy variables for NewExist: is_new_business (1 if NewExist == 1 else 0)
# - Create dummy variables for UrbanRural: is_urban (1 if UrbanRural == 1 else 0)
# - Consider interaction NewExist × UrbanRural if sample size allows

# Leakage concern:
# - No direct leakage in these variables.
# - However, "Unknown" categories (NewExist = 0, UrbanRural = 0) may have very different default behavior,
#   possibly because missingness correlates with loan performance. We keep them as separate categories.

# Lending interpretation:
# - Shorter‑term loans appear riskier → banks should require stronger collateral or higher rates for short terms.
# - Existing businesses (NewExist = 2) default more than new businesses — counter‑intuitive.
#   Possibly because existing businesses take larger or riskier loans.
# - Urban loans default more than rural → banks may need stricter underwriting in cities, or charge higher rates.


#%% 6C. Part 1 EDA — Owner: Huyen Anh

# Write your EDA code here.
# Focus idea: patterns that may require nonlinear models or interactions.
# Example only:
# huyen_anh_revline = df_model.groupby("RevLineCr")["y"].agg(["count", "mean"]).sort_values("mean", ascending=False)
# print(huyen_anh_revline.head(15))


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
df_model["log_DisbursementGross"] = np.log1p(df_model["DisbursementGross"])

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
    "log_DisbursementGross",   
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
#Term, NoEmp, GrAppv, SBA_Appv – Core loan characteristics with clear EDA differences

#Portion, unguaranteed_ratio – Bank's own risk exposure affects screening quality

#log transforms – Linear models need normality; logs fix skewness in amounts and employee counts

#Recession, approval_year – Macro/time controls

#RealEstate, same_state_bank – Simple proxies for collateral quality and information advantage

#Simple interactions – Logistic needs explicit interactions; trees find them automatically

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
    # 2. Which C values did you try? Currently tried: C = 1.0 (Ridge), C = 0.5 (Lasso & ElasticNet)
    # 3. Did the solver converge? Check n_iter_.
    # 4. For LDA/QDA, did regularization help?
# LDA: No regularization needed. Works well out-of-box.
#          Profit: 50.13M, slightly below Ridge.
#    
#    QDA: Regularization (reg_param=0.1) helped vs default QDA,
#         but still underperforms LDA and all logistic models.
#         Reason: QDA assumes different covariance per class,
#         but with many categorical features (one-hot encoded),
#         covariance estimation becomes unstable.
 

   # Write your logistic / discriminant models here.
    # Use preprocess_scaled.
    #
    # Example result names you can use:
    # ridge_tuned, lasso_tuned, elastic_tuned, lda_tuned, qda_best
    pass

# Hai Anh notes:
# Best logistic result: Ridge_L2 (C=1.0, penalty='l2', solver='lbfgs')
#   - Validation profit: 50,457,678
#   - Optimal threshold: 0.223 (22.3% default probability)
#   - Approval rate: 71.7%
#   - Approved default rate: 6.3%
#   - AUC: 0.840

# Best LDA/QDA result: LDA (no regularization needed)
#   - Validation profit: 50,133,917
#   - Slightly lower profit than Ridge
#   - Lower approval rate (67.6%) means more loans denied

# Solver/convergence concern:
#   - Ridge (lbfgs): converged, n_iter_ ~ 15-20 iterations, fast (1.56s)
#   - Lasso & ElasticNet (saga): converged but very slow (443s, 276s)
#   - Saga solver requires more iterations (max_iter=3000 was sufficient)
#   - For production, Ridge is preferred due to speed

# Interpretation for the bank:
#   1. Ridge logistic regression is the best linear model
#   2. At optimal threshold (22.3%), approve 71.7% of loan applications
#   3. Only 6.3% of approved loans default (vs 17.1% baseline default rate)
#   4. This policy generates ~50.5M profit on validation set
#   5. Lasso/ElasticNet not worth the extra computation time
#   6. LDA is a good fast alternative but approves fewer loans
#   7. QDA underperforms due to violation of normality assumptions


#%% 14D. Neural network workspace — Owner: Huyen Anh



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
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    # Goal:
    # Run CV for logistic settings.
    # LDA/QDA can be checked separately if memory becomes heavy.

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    categorical_transformer_dense = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    
    preprocess_dense = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer_scaled, numeric_cols),
            ("cat", categorical_transformer_dense, categorical_cols),
        ]
    )
    
    # ========== LOGISTIC REGRESSION  ==========
    print("\n=== Logistic Regression CV ===")
    
    param_grid = [
        {"penalty": "l2", "C": 0.1, "solver": "lbfgs"},
        {"penalty": "l2", "C": 1.0, "solver": "lbfgs"},
        {"penalty": "l2", "C": 10.0, "solver": "lbfgs"},
        {"penalty": "l1", "C": 0.1, "solver": "saga"},
        {"penalty": "l1", "C": 0.5, "solver": "saga"},
        {"penalty": "l1", "C": 1.0, "solver": "saga"},
        {"penalty": "elasticnet", "C": 0.5, "l1_ratio": 0.5, "solver": "saga"},
        {"penalty": "elasticnet", "C": 1.0, "l1_ratio": 0.7, "solver": "saga"},
    ]
    
    for params in param_grid:
        kwargs = {
            "penalty": params["penalty"],
            "C": params["C"],
            "solver": params["solver"],
            "max_iter": 3000,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        if params["penalty"] == "elasticnet":
            kwargs["l1_ratio"] = params["l1_ratio"]
        
        # Logistic có thể dùng preprocess_scaled (sparse OK)
        candidate = Pipeline(steps=[
            ("preprocess", preprocess_scaled),
            ("model", LogisticRegression(**kwargs)),
        ])
        
        try:
            scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            add_cv_result("Hai Anh", "Logistic", params, scores)
            print(f"{params['penalty']}, C={params['C']}, l1_ratio={params.get('l1_ratio', 'N/A')} -> AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            print(f"Failed: {params} - {str(e)[:100]}")
    
    # ========== LDA  ==========
    print("\n=== LDA CV ===")
    
    lda = Pipeline(steps=[
        ("preprocess", preprocess_dense),
        ("model", LinearDiscriminantAnalysis())
    ])
    
    try:
        lda_scores = cross_val_score(lda, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        add_cv_result("Hai Anh", "LDA", {"reg_param": 0}, lda_scores)
        print(f"LDA -> AUC: {lda_scores.mean():.4f} (+/- {lda_scores.std():.4f})")
    except Exception as e:
        print(f"LDA failed: {e}")
    
    # ========== QDA (dùng preprocess_dense) ==========
    print("\n=== QDA CV ===")
    
    qda = Pipeline(steps=[
        ("preprocess", preprocess_dense),
        ("model", QuadraticDiscriminantAnalysis(reg_param=0.1))
    ])
    
    try:
        qda_scores = cross_val_score(qda, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        add_cv_result("Hai Anh", "QDA", {"reg_param": 0.1}, qda_scores)
        print(f"QDA -> AUC: {qda_scores.mean():.4f} (+/- {qda_scores.std():.4f})")
    except Exception as e:
        print(f"QDA failed: {e}")

RUN_LOGIT_DA_CV = False
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
 
# Hai Anh CV notes:
# Best logistic CV setting:
# Did Lasso/ElasticNet improve over Ridge?

# Best logistic CV setting:
#   - Ridge (L2 penalty) with C=10.0, solver='lbfgs'
#   - CV AUC: 0.8550 (+/- 0.0011)
#   - C=1.0 and C=10.0 are very close (0.8549 vs 0.8550)
#   - Validation profit earlier showed C=1.0 best (50.46M)
#   - Both are acceptable; C=1.0 is safer to avoid overfitting

# Did Lasso/ElasticNet improve over Ridge?
#   - NO. Ridge consistently outperforms Lasso/ElasticNet
#   - Lasso/ElasticNet also have convergence issues (max_iter reached)
#   - Saga solver slower and less stable than lbfgs

# LDA performance:
#   - CV AUC: 0.8401 (lower than Ridge's 0.8550)
#   - Confirms validation profit result: LDA < Ridge

# QDA performance:
#   - CV AUC: 0.8123 (lowest)
#   - Collinearity warning due to many one-hot encoded features
#   - Not recommended for this dataset

# Recommendation:
#   - Use Ridge (L2, C=1.0 or C=10.0) with solver='lbfgs'
#   - Fast convergence, no warnings, best AUC and profit

#%% 15D. Neural network cross-validation / tuning — Owner: Huyen Anh


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
## TOP 3 Logistic settings worth validating:
# 1. Ridge L2, C=10.0, solver='lbfgs'  → AUC: 0.85496
# 2. Ridge L2, C=1.0, solver='lbfgs'   → AUC: 0.85486
# 3. Ridge L2, C=0.1, solver='lbfgs'   → AUC: 0.85454
#
# Lasso and ElasticNet have lower AUC and convergence warnings → skip
#
# LDA: AUC 0.84013 → good but lower than Ridge → keep as baseline
#
# QDA: AUC 0.81227 → too low, collinearity issues → skip

# Recommendation for final model:
# - Ridge L2, C=1.0 (or C=10.0) with solver='lbfgs'
# - Already validated in Block 14C with profit 50.46M
# - Fast training, no convergence issues, best overall performance
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





#%% 21. Chạy thêm


#%% 14C Extended – Tuning Logistic/LDA/QDA (Hai Anh)

RUN_TUNING_EXTENDED = True

if RUN_TUNING_EXTENDED:
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    import pandas as pd
    import numpy as np
    
    # Storage for all results
    extended_results = []
    
    # ========== PREPROCESSING DENSE CHO LDA/QDA ==========
    categorical_transformer_dense = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    
    preprocess_dense = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer_scaled, numeric_cols),
            ("cat", categorical_transformer_dense, categorical_cols),
        ]
    )
    
    # ========== 1. RIDGE with multiple C ==========
    print("\n=== Ridge Logistic (L2) ===")
    
    C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    for C in C_values:
        ridge = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=3000, random_state=RANDOM_STATE)
        ridge_pipe, ridge_prob, ridge_theory, ridge_tuned, ridge_table = fit_predict_evaluate(
            f"Ridge_C_{C}", ridge, preprocess_scaled
        )
        extended_results.append(ridge_tuned)
        print(f"C={C}: Profit={ridge_tuned['net_profit']:,.0f}, AUC={ridge_tuned['auc']:.4f}, "
              f"Threshold={ridge_tuned['threshold_default']:.4f}, Approval Rate={ridge_tuned['approval_rate']:.3f}")
    
    # ========== 2. LASSO – Feature Selection ==========
    print("\n=== Lasso Logistic (L1) ===")
    
    C_lasso = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    for C in C_lasso:
        lasso = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=5000, random_state=RANDOM_STATE)
        lasso_pipe, lasso_prob, lasso_theory, lasso_tuned, lasso_table = fit_predict_evaluate(
            f"Lasso_C_{C}", lasso, preprocess_scaled
        )
        extended_results.append(lasso_tuned)
        print(f"C={C}: Profit={lasso_tuned['net_profit']:,.0f}, AUC={lasso_tuned['auc']:.4f}, "
              f"Threshold={lasso_tuned['threshold_default']:.4f}")
        
        # In số feature được chọn (non-zero coefficients)
        coefs = lasso_pipe.named_steps['model'].coef_[0]
        n_selected = np.sum(np.abs(coefs) > 1e-6)
        print(f"  → Non-zero coefficients: {n_selected} / {len(coefs)}")
    
    # ========== 3. ELASTICNET – Mixed ==========
    print("\n=== ElasticNet Logistic ===")
    
    elastic_configs = [
        {"C": 0.1, "l1_ratio": 0.9, "name": "EN_almost_lasso"},
        {"C": 0.5, "l1_ratio": 0.5, "name": "EN_mix"},
        {"C": 1.0, "l1_ratio": 0.1, "name": "EN_almost_ridge"},
        {"C": 0.5, "l1_ratio": 0.7, "name": "EN_more_lasso"},
        {"C": 2.0, "l1_ratio": 0.3, "name": "EN_more_ridge"},
    ]
    
    for cfg in elastic_configs:
        elastic = LogisticRegression(
            penalty="elasticnet", 
            C=cfg["C"], 
            l1_ratio=cfg["l1_ratio"],
            solver="saga", 
            max_iter=5000, 
            random_state=RANDOM_STATE
        )
        elastic_pipe, elastic_prob, elastic_theory, elastic_tuned, elastic_table = fit_predict_evaluate(
            cfg["name"], elastic, preprocess_scaled
        )
        extended_results.append(elastic_tuned)
        print(f"{cfg['name']} (C={cfg['C']}, l1_ratio={cfg['l1_ratio']}): "
              f"Profit={elastic_tuned['net_profit']:,.0f}, AUC={elastic_tuned['auc']:.4f}")
    
    # ========== 4. LDA vs SHRINKAGE ==========
    print("\n=== LDA with Shrinkage ===")
    
    shrinkage_values = [0.0, 0.1, 0.2, 0.5, 0.8, 0.99, 'auto']
    
    for shrinkage in shrinkage_values:
        # Quan trọng: set solver='lsqr' hoặc 'eigen' để dùng shrinkage
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
        lda_pipe, lda_prob, lda_theory, lda_tuned, lda_table = fit_predict_evaluate(
            f"LDA_shrink_{shrinkage}", lda, preprocess_dense
        )
        extended_results.append(lda_tuned)
        print(f"shrinkage={shrinkage}: Profit={lda_tuned['net_profit']:,.0f}, AUC={lda_tuned['auc']:.4f}, "
              f"Approval Rate={lda_tuned['approval_rate']:.3f}")
    
    # ========== 5. QDA vs REGULARIZATION ==========
    print("\n=== QDA with Regularization ===")
    
    reg_params = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    
    for reg in reg_params:
        qda = QuadraticDiscriminantAnalysis(reg_param=reg)
        qda_pipe, qda_prob, qda_theory, qda_tuned, qda_table = fit_predict_evaluate(
            f"QDA_reg_{reg}", qda, preprocess_dense
        )
        extended_results.append(qda_tuned)
        print(f"reg_param={reg}: Profit={qda_tuned['net_profit']:,.0f}, AUC={qda_tuned['auc']:.4f}")
    
  
    print("\n" + "="*80)
    print("=== EXTENDED TUNING RESULTS (sorted by net_profit) ===")
    print("="*80)
    
    extended_df = pd.DataFrame(extended_results)
    extended_df = extended_df.sort_values("net_profit", ascending=False)
    
    # Chỉ giữ các cột quan trọng
    result_cols = ["model", "threshold_default", "auc", "brier", "net_profit", 
                   "approval_rate", "approved_default_rate"]
    print(extended_df[result_cols].head(20))
    
   
    # extended_df.to_csv("logit_lda_qda_tuning_results.csv", index=False)

RUN_TUNING_EXTENDED = False

#Lasso with C=0.01 , Profit: 50.75M (beats Ridge by ~0.24M)
#Ridge performs consistently well. Best Ridge: C=2.0 (50.51M), C=20.0 (50.50M)
# ElasticNet 50.57M
#LDA best at shrinkage=0.0, Profit: 50.13M
#QDA, 44.57M




















