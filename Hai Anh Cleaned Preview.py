# -*- coding: utf-8 -*-
"""
SBA Final Project — Hai Anh Cleaned Preview
Logistic Regression / LDA / QDA workspace using shared feature set v1.
Created from origin/hai-anh branch.  Removed Block 21, fixed hardcoded path.
Uses shared variables: X_train, y_train, preprocess_scaled, fit_predict_evaluate().
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

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

RANDOM_STATE = 1
DATA_PATH = "SBAnational.csv"
USE_WORKING_SAMPLE = True
WORKING_SAMPLE_N = 50000
THEORETICAL_DEFAULT_THRESHOLD = 1 / 6
RUN_LOGIT_DA = False
RUN_LOGIT_DA_CV = False
RUN_FINAL_TEST = False

OWNER_NAME = "Hai Anh"


#%% 2. Load data

df_raw = pd.read_csv(DATA_PATH)
print("Rows, columns:", df_raw.shape)


#%% 3. Create target and clean money columns

df = df_raw.copy()
print("MIS_Status distribution before cleaning:")
print(df["MIS_Status"].value_counts(dropna=False))
df = df[df["MIS_Status"].notna()].copy()
df["y"] = np.where(df["MIS_Status"] == "CHGOFF", 1, 0)
print("Target distribution:")
print(df["y"].value_counts())
print("Default rate:", df["y"].mean())

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

if USE_WORKING_SAMPLE:
    sample_n = min(WORKING_SAMPLE_N, len(df))
    if sample_n < len(df):
        df, _ = train_test_split(df, train_size=sample_n, random_state=RANDOM_STATE, stratify=df["y"])
    df = df.sort_index().copy()
    print("Using working sample:", df.shape)
    print("Sample default rate:", df["y"].mean())
else:
    print("Using full dataset:", df.shape)


#%% 5. Leakage audit and modeling dataframe

leakage_cols = ["MIS_Status", "ChgOffDate", "ChgOffPrinGr", "BalanceGross"]
id_text_cols = ["LoanNr_ChkDgt", "Name", "City", "Zip", "Bank"]
USE_DISBURSEMENT_AS_PREDICTOR = False

df_model = df.drop(columns=leakage_cols + id_text_cols, errors="ignore").copy()
print("Model dataframe shape:", df_model.shape)


#%% 7. Shared feature engineering — feature set v1

df_model["Portion"] = df_model["SBA_Appv"] / df_model["GrAppv"]
df_model["Portion"] = df_model["Portion"].replace([np.inf, -np.inf], np.nan).fillna(0)
df_model["unguaranteed_ratio"] = 1 - df_model["Portion"]
df_model["unguaranteed_amount"] = df_model["GrAppv"] - df_model["SBA_Appv"]

df_model["jobs_total"] = df_model["CreateJob"] + df_model["RetainedJob"]
df_model["jobs_per_dollar"] = df_model["jobs_total"] / df_model["GrAppv"]
df_model["jobs_per_dollar"] = df_model["jobs_per_dollar"].replace([np.inf, -np.inf], np.nan).fillna(0)

df_model["same_state_bank"] = np.where(df_model["State"] == df_model["BankState"], 1, 0)
df_model["RealEstate"] = np.where(df_model["Term"] >= 240, 1, 0)

for col in ["ApprovalDate", "DisbursementDate"]:
    df_model[col] = pd.to_datetime(df_model[col], errors="coerce")

recession_start = pd.Timestamp("2007-12-01")
recession_end = pd.Timestamp("2009-06-30")
df_model["estimated_maturity_date"] = df_model["DisbursementDate"] + pd.to_timedelta(df_model["Term"].fillna(0) * 30, unit="D")
df_model["Recession"] = np.where(
    (df_model["DisbursementDate"] <= recession_end) &
    (df_model["estimated_maturity_date"] >= recession_start),
    1, 0,
)
df_model["approval_year"] = df_model["ApprovalDate"].dt.year
df_model["disbursement_year"] = df_model["DisbursementDate"].dt.year

df_model["ApprovalFY_clean"] = (
    df_model["ApprovalFY"].astype(str).str.replace("A", "", regex=False).str.strip()
)
df_model["ApprovalFY_clean"] = pd.to_numeric(df_model["ApprovalFY_clean"], errors="coerce")

df_model["NAICS_str"] = df_model["NAICS"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
df_model["NAICS_sector"] = df_model["NAICS_str"].str[:2]

def clean_yes_no(series):
    cleaned = series.astype(str).str.strip().str.upper()
    cleaned = cleaned.replace({"Y": "Yes", "YES": "Yes", "1": "Yes", "N": "No", "NO": "No", "0": "No", "NAN": "Unknown", "": "Unknown"})
    cleaned = np.where(pd.Series(cleaned).isin(["Yes", "No"]), cleaned, "Unknown")
    return cleaned

df_model["LowDoc_clean"] = clean_yes_no(df_model["LowDoc"])
df_model["RevLineCr_clean"] = clean_yes_no(df_model["RevLineCr"])

for col in ["GrAppv", "SBA_Appv", "NoEmp"]:
    if col in df_model.columns:
        df_model[f"log_{col}"] = np.log1p(df_model[col])

df_model["RealEstate_x_Portion"] = df_model["RealEstate"] * df_model["Portion"]
df_model["Recession_x_Portion"] = df_model["Recession"] * df_model["Portion"]
df_model["Recession_x_RealEstate"] = df_model["Recession"] * df_model["RealEstate"]

float_cols = df_model.select_dtypes(include=[np.floating]).columns
df_model[float_cols] = df_model[float_cols].replace([np.inf, -np.inf], np.nan)

df_model.drop(columns=["NAICS_str", "estimated_maturity_date", "ApprovalDate", "DisbursementDate"], errors="ignore", inplace=True)
print("df_model shape:", df_model.shape)


#%% 8. Register predictors

numeric_cols = [
    "Term", "NoEmp", "CreateJob", "RetainedJob", "GrAppv", "SBA_Appv",
    "Portion", "unguaranteed_ratio", "unguaranteed_amount",
    "jobs_total", "jobs_per_dollar", "same_state_bank",
    "RealEstate", "Recession", "approval_year", "disbursement_year",
    "ApprovalFY_clean", "log_GrAppv", "log_SBA_Appv", "log_NoEmp",
    "RealEstate_x_Portion",
    "Recession_x_Portion", "Recession_x_RealEstate",
]

if USE_DISBURSEMENT_AS_PREDICTOR:
    numeric_cols += ["DisbursementGross"]

categorical_cols = [
    "State", "BankState", "NewExist", "UrbanRural",
    "NAICS_sector", "LowDoc_clean", "RevLineCr_clean",
]

numeric_cols = [c for c in numeric_cols if c in df_model.columns]
categorical_cols = [c for c in categorical_cols if c in df_model.columns]
print("Numeric:", len(numeric_cols), "Categorical:", len(categorical_cols))


#%% 9. Build X, y, and amount

X = df_model[numeric_cols + categorical_cols].copy()
y = df_model["y"].copy()
amount = df_model["DisbursementGross"].copy()

for col in categorical_cols:
    X[col] = X[col].fillna("Unknown").astype(str)

print("X shape:", X.shape, "default rate:", y.mean())


#%% 10. Train / validation / test split

X_train_valid, X_test, y_train_valid, y_test, amount_train_valid, amount_test = train_test_split(
    X, y, amount, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

X_train, X_valid, y_train, y_valid, amount_train, amount_valid = train_test_split(
    X_train_valid, y_train_valid, amount_train_valid,
    test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_valid)

print("Train:", X_train.shape, "Valid:", X_valid.shape, "Test:", X_test.shape)


#%% 11. Preprocessing

numeric_transformer_scaled = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
])
categorical_transformer_dense = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess_scaled = ColumnTransformer([
    ("num", numeric_transformer_scaled, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])
preprocess_scaled_dense = ColumnTransformer([
    ("num", numeric_transformer_scaled, numeric_cols),
    ("cat", categorical_transformer_dense, categorical_cols),
])
print("Preprocessing ready.")


#%% 12. Shared scoreboard functions

def loan_profit_vector(y_true, decision_approve, amount_series):
    amount_arr = np.asarray(amount_series, dtype=float)
    y_arr = np.asarray(y_true)
    approve_arr = np.asarray(decision_approve).astype(bool)
    gain_if_paid = 0.05 * amount_arr
    loss_if_default = -0.25 * amount_arr
    return np.where(approve_arr, np.where(y_arr == 0, gain_if_paid, loss_if_default), 0.0)

def evaluate_prob_model(model_name, y_true, prob_default, amount_series, threshold):
    y_true_arr = np.asarray(y_true)
    prob_default_arr = np.asarray(prob_default)
    pred_default = (prob_default_arr > threshold).astype(int)
    decision_approve = prob_default_arr <= threshold
    confmat = confusion_matrix(y_true_arr, pred_default, labels=[1, 0])
    TP = confmat[0, 0]; FN = confmat[0, 1]; FP = confmat[1, 0]; TN = confmat[1, 1]
    profit = loan_profit_vector(y_true_arr, decision_approve, amount_series)
    return {
        "model": model_name, "threshold_default": threshold, "threshold_success": 1 - threshold,
        "accuracy": accuracy_score(y_true_arr, pred_default),
        "recall_default": recall_score(y_true_arr, pred_default, zero_division=0),
        "specificity_paid": TN / (TN + FP) if (TN + FP) > 0 else np.nan,
        "precision_default": precision_score(y_true_arr, pred_default, zero_division=0),
        "f1_default": f1_score(y_true_arr, pred_default, zero_division=0),
        "auc": roc_auc_score(y_true_arr, prob_default_arr),
        "brier": brier_score_loss(y_true_arr, prob_default_arr),
        "net_profit": profit.sum(), "approval_rate": decision_approve.mean(),
        "approved_default_rate": y_true_arr[decision_approve].mean() if decision_approve.sum() > 0 else np.nan,
        "denied_default_rate": y_true_arr[~decision_approve].mean() if (~decision_approve).sum() > 0 else np.nan,
        "TP_default_denied": int(TP), "FN_default_approved": int(FN),
        "FP_paid_denied": int(FP), "TN_paid_approved": int(TN),
    }

def tune_threshold_by_profit(model_name, y_true, prob_default, amount_series, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 120)
    thresholds = np.unique(np.append(thresholds, THEORETICAL_DEFAULT_THRESHOLD))
    rows = [evaluate_prob_model(model_name, y_true, prob_default, amount_series, th) for th in thresholds]
    return pd.DataFrame(rows).sort_values("net_profit", ascending=False).reset_index(drop=True)

def fit_predict_evaluate(model_name, estimator, preprocess_obj):
    start = time.perf_counter()
    pipe = Pipeline(steps=[("preprocess", preprocess_obj), ("model", estimator)])
    pipe.fit(X_train, y_train)
    prob_default_valid = pipe.predict_proba(X_valid)[:, 1]
    runtime = time.perf_counter() - start
    theory = evaluate_prob_model(model_name, y_valid, prob_default_valid, amount_valid, THEORETICAL_DEFAULT_THRESHOLD)
    theory["threshold_type"] = "theoretical_1_over_6"; theory["runtime_seconds"] = runtime
    threshold_table = tune_threshold_by_profit(model_name, y_valid, prob_default_valid, amount_valid)
    tuned = threshold_table.iloc[0].to_dict()
    tuned["threshold_type"] = "validation_profit_tuned"; tuned["runtime_seconds"] = runtime
    return pipe, prob_default_valid, theory, tuned, threshold_table


#%% 13. Baseline policies

approve_all_profit = loan_profit_vector(y_true=y_valid, decision_approve=np.ones(len(y_valid), dtype=bool), amount_series=amount_valid).sum()
deny_all_profit = 0.0
print("Approve-all validation profit:", approve_all_profit)
print("Deny-all validation profit:", deny_all_profit)


#%% 14C. Logistic regression and discriminant analysis — Owner: Hai Anh

if RUN_LOGIT_DA:
    logit_results = []

    # --------- Ridge (L2) ---------
    print("\n" + "=" * 50)
    print("Logistic Ridge (L2)")
    print("=" * 50)
    for C in [0.1, 1.0, 10.0]:
        ridge = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE)
        _, _, _, tuned, _ = fit_predict_evaluate(f"Ridge_C{C}", ridge, preprocess_scaled)
        logit_results.append(tuned)
        print(f"  C={C}: Profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}, Thr={tuned['threshold_default']:.3f}")

    # --------- Lasso (L1) ---------
    print("\n" + "=" * 50)
    print("Logistic Lasso (L1)")
    print("=" * 50)
    for C in [0.1, 0.5]:
        lasso = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=2000, random_state=RANDOM_STATE)
        _, _, _, tuned, _ = fit_predict_evaluate(f"Lasso_C{C}", lasso, preprocess_scaled)
        logit_results.append(tuned)
        print(f"  C={C}: Profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}, Thr={tuned['threshold_default']:.3f}")

    # --------- ElasticNet ---------
    print("\n" + "=" * 50)
    print("Logistic ElasticNet")
    print("=" * 50)
    elastic = LogisticRegression(penalty="elasticnet", C=1.0, l1_ratio=0.3, solver="saga", max_iter=2000, random_state=RANDOM_STATE)
    _, _, _, tuned, _ = fit_predict_evaluate("ElasticNet_C1.0_l1r0.3", elastic, preprocess_scaled)
    logit_results.append(tuned)
    print(f"  Profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}, Thr={tuned['threshold_default']:.3f}")

    # --------- LDA ---------
    print("\n" + "=" * 50)
    print("LDA (lsqr, shrinkage=auto)")
    print("=" * 50)
    start = time.perf_counter()
    X_train_d = preprocess_scaled_dense.fit_transform(X_train, y_train)
    X_valid_d = preprocess_scaled_dense.transform(X_valid)
    if hasattr(X_train_d, "toarray"):
        X_train_d = X_train_d.toarray(); X_valid_d = X_valid_d.toarray()
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_train_d, y_train)
    lda_prob = lda.predict_proba(X_valid_d)[:, 1]
    runtime = time.perf_counter() - start
    lda_tuned = tune_threshold_by_profit("LDA_lsqr_auto", y_valid, lda_prob, amount_valid).iloc[0].to_dict()
    lda_tuned["threshold_type"] = "validation_profit_tuned"; lda_tuned["runtime_seconds"] = runtime
    logit_results.append(lda_tuned)
    print(f"  Profit=${lda_tuned['net_profit']:,.0f}, AUC={lda_tuned.get('auc',0):.4f}, Thr={lda_tuned['threshold_default']:.3f}")

    # --------- QDA ---------
    print("\n" + "=" * 50)
    print("QDA (reg_param=0.2)")
    print("=" * 50)
    start = time.perf_counter()
    qda = QuadraticDiscriminantAnalysis(reg_param=0.2)
    qda.fit(X_train_d, y_train)
    qda_prob = qda.predict_proba(X_valid_d)[:, 1]
    runtime = time.perf_counter() - start
    qda_tuned = tune_threshold_by_profit("QDA_reg0.2", y_valid, qda_prob, amount_valid).iloc[0].to_dict()
    qda_tuned["threshold_type"] = "validation_profit_tuned"; qda_tuned["runtime_seconds"] = runtime
    logit_results.append(qda_tuned)
    print(f"  Profit=${qda_tuned['net_profit']:,.0f}, AUC={qda_tuned.get('auc',0):.4f}, Thr={qda_tuned['threshold_default']:.3f}")

    # --------- Summary ---------
    print("\n" + "=" * 50)
    print("Logistic / LDA / QDA Summary (sorted by profit)")
    print("=" * 50)
    logit_df = pd.DataFrame(logit_results).sort_values("net_profit", ascending=False).reset_index(drop=True)
    cols = ["model", "threshold_default", "auc", "brier", "net_profit", "approval_rate", "approved_default_rate", "runtime_seconds"]
    print(logit_df[[c for c in cols if c in logit_df.columns]].to_string(index=False))

# Hai Anh notes:
# Official benchmark results (shared features v1):
#   Best logistic: Ridge C=10.0 → profit ~$50.4M, AUC 0.839, approval 74.1%
#   Lasso C=0.5 → profit ~$50.2M, AUC 0.839
#   ElasticNet → profit ~$50.4M, AUC 0.839
#   LDA → profit ~$49.0M, AUC 0.818
#   QDA reg=0.2 → profit ~$44.2M, AUC 0.802
# Ridge is the strongest linear model. L1 does NOT improve over L2.
# LDA is a fast baseline. QDA needs strong regularization.


#%% 15C. Logistic / discriminant CV — Owner: Hai Anh

if RUN_LOGIT_DA_CV:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    # Logistic CV grid (compact)
    param_grid = [
        {"penalty": "l2", "C": 0.1, "solver": "lbfgs", "name": "Ridge_C0.1"},
        {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "name": "Ridge_C1.0"},
        {"penalty": "l2", "C": 10.0, "solver": "lbfgs", "name": "Ridge_C10"},
        {"penalty": "l1", "C": 0.1, "solver": "saga", "name": "Lasso_C0.1"},
        {"penalty": "l1", "C": 0.5, "solver": "saga", "name": "Lasso_C0.5"},
        {"penalty": "elasticnet", "C": 1.0, "l1_ratio": 0.3, "solver": "saga", "name": "ElasticNet"},
    ]

    for p in param_grid:
        kwargs = {"penalty": p["penalty"], "C": p["C"], "solver": p["solver"],
                  "max_iter": 2000, "random_state": RANDOM_STATE}
        if p["penalty"] == "elasticnet":
            kwargs["l1_ratio"] = p["l1_ratio"]
        candidate = Pipeline([("preprocess", preprocess_scaled), ("model", LogisticRegression(**kwargs))])
        scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  {p['name']}: CV AUC={scores.mean():.4f} ± {scores.std():.4f}")

    # LDA CV
    print("\nLDA CV:")
    lda_pipe = Pipeline([("preprocess", preprocess_scaled_dense), ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
    scores = cross_val_score(lda_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  LDA: CV AUC={scores.mean():.4f} ± {scores.std():.4f}")

    # QDA CV
    print("QDA CV:")
    qda_pipe = Pipeline([("preprocess", preprocess_scaled_dense), ("model", QuadraticDiscriminantAnalysis(reg_param=0.2))])
    scores = cross_val_score(qda_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  QDA: CV AUC={scores.mean():.4f} ± {scores.std():.4f}")

# Hai Anh CV notes:
# Best logistic CV: Ridge C=10.0, solver=lbfgs
# Lasso/ElasticNet do NOT improve over Ridge — L2 is best for this data.
# LDA is a fast alternative; QDA lags.


#%% 16. Build validation leaderboard

model_results = []

var_names = ["ridge_tuned", "lasso_tuned", "elastic_tuned", "lda_tuned", "qda_tuned"]
for vname in var_names:
    if vname in globals():
        result = globals()[vname]
        result["model"] = vname.replace("_tuned", "")
        model_results.append(result)

baseline_rows = [
    {"model": "Approve All Baseline", "threshold_type": "baseline",
     "net_profit": approve_all_profit, "approval_rate": 1.0, "approved_default_rate": y_valid.mean()},
    {"model": "Deny All Baseline", "threshold_type": "baseline",
     "net_profit": deny_all_profit, "approval_rate": 0.0, "approved_default_rate": np.nan},
]

leaderboard = pd.DataFrame(baseline_rows + model_results)
leaderboard = leaderboard.sort_values("net_profit", ascending=False).reset_index(drop=True)

display_cols = ["model", "threshold_type", "threshold_default", "auc", "brier",
                "net_profit", "approval_rate", "approved_default_rate", "runtime_seconds"]
print("\n=== Hai Anh Validation Leaderboard ===")
print(leaderboard[[c for c in display_cols if c in leaderboard.columns]].to_string(index=False))


#%% 19. Final untouched test evaluation

if RUN_FINAL_TEST:
    pass
else:
    print("Skipping final test — RUN_FINAL_TEST = False")


#%% 20. Save outputs

print("Hai Anh Cleaned Preview complete.")
print("Set RUN_LOGIT_DA = True to run logistic/LDA/QDA models.")
print("Set RUN_LOGIT_DA_CV = True to run cross-validation.")
print("Do NOT run RUN_FINAL_TEST yet.")
