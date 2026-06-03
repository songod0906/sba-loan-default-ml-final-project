# -*- coding: utf-8 -*-
"""
SBA Final Project — Hai An Shared V1 Rerun
KNN + Tree-family (DT/Bagging/RF/AdaBoost/HGB) with FULL shared feature set v1.
Upgraded from origin/haian branch.  Blocks 7-8 replaced with 24 numeric + 7 categorical.
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
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, brier_score_loss,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_HIST_GB = True
except ImportError:
    _HAS_HIST_GB = False

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

RANDOM_STATE = 1
DATA_PATH = "SBAnational.csv"
USE_WORKING_SAMPLE = True
WORKING_SAMPLE_N = 50000
THEORETICAL_DEFAULT_THRESHOLD = 1 / 6

OWNER_NAME = "Hai An"

RUN_KNN = False
RUN_TREE_MODELS = False
RUN_FINAL_TEST = False


#%% 2. Load data

df_raw = pd.read_csv(DATA_PATH)
print("Rows, columns:", df_raw.shape)


#%% 3. Create target and clean money columns

df = df_raw.copy()
print("MIS_Status distribution before cleaning:")
print(df["MIS_Status"].value_counts(dropna=False))
df = df[df["MIS_Status"].notna()].copy()
df["y"] = np.where(df["MIS_Status"] == "CHGOFF", 1, 0)
print("Target distribution:"); print(df["y"].value_counts())
print("Default rate:", df["y"].mean())

money_cols = ["DisbursementGross", "GrAppv", "SBA_Appv", "BalanceGross", "ChgOffPrinGr"]
for col in money_cols:
    if col in df.columns:
        df[col] = (df[col].astype(str).str.replace("$", "", regex=False)
                   .str.replace(",", "", regex=False).str.strip()
                   .replace({"nan": np.nan, "": np.nan}).astype(float))


#%% 4. Optional working sample

if USE_WORKING_SAMPLE:
    sample_n = min(WORKING_SAMPLE_N, len(df))
    if sample_n < len(df):
        df, _ = train_test_split(df, train_size=sample_n, random_state=RANDOM_STATE, stratify=df["y"])
    df = df.sort_index().copy()
    print("Using working sample:", df.shape, "default rate:", df["y"].mean())


#%% 5. Leakage audit and modeling dataframe

leakage_cols = ["MIS_Status", "ChgOffDate", "ChgOffPrinGr", "BalanceGross"]
id_text_cols = ["LoanNr_ChkDgt", "Name", "City", "Zip", "Bank"]
USE_DISBURSEMENT_AS_PREDICTOR = False
df_model = df.drop(columns=leakage_cols + id_text_cols, errors="ignore").copy()
print("Model dataframe shape:", df_model.shape)


#%% 7. Shared feature engineering — FULL feature set v1

# 7.1 SBA guarantee
df_model["Portion"] = df_model["SBA_Appv"] / df_model["GrAppv"]
df_model["unguaranteed_ratio"] = 1 - df_model["Portion"]
df_model["unguaranteed_amount"] = df_model["GrAppv"] - df_model["SBA_Appv"]

# 7.2 Job impact
df_model["jobs_total"] = df_model["CreateJob"] + df_model["RetainedJob"]
df_model["jobs_per_dollar"] = df_model["jobs_total"] / df_model["GrAppv"]

# 7.3 Same-state lender
df_model["same_state_bank"] = np.where(df_model["State"] == df_model["BankState"], 1, 0)

# 7.4 Real-estate proxy
df_model["RealEstate"] = np.where(df_model["Term"] >= 240, 1, 0)

# 7.5 Date features + Recession
for col in ["ApprovalDate", "DisbursementDate"]:
    df_model[col] = pd.to_datetime(df_model[col], errors="coerce")

recession_start = pd.Timestamp("2007-12-01")
recession_end = pd.Timestamp("2009-06-30")
df_model["estimated_maturity_date"] = df_model["DisbursementDate"] + pd.to_timedelta(df_model["Term"].fillna(0) * 30, unit="D")
df_model["Recession"] = np.where(
    (df_model["DisbursementDate"] <= recession_end) &
    (df_model["estimated_maturity_date"] >= recession_start), 1, 0)

df_model["approval_year"] = df_model["ApprovalDate"].dt.year
df_model["disbursement_year"] = df_model["DisbursementDate"].dt.year

# ApprovalFY_clean
df_model["ApprovalFY_clean"] = (
    df_model["ApprovalFY"].astype(str).str.replace("A", "", regex=False).str.strip())
df_model["ApprovalFY_clean"] = pd.to_numeric(df_model["ApprovalFY_clean"], errors="coerce")

# Drop temp columns
df_model.drop(columns=["ApprovalDate", "DisbursementDate", "estimated_maturity_date"], errors="ignore", inplace=True)

# 7.6 NAICS sector
df_model["NAICS_str"] = df_model["NAICS"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
df_model["NAICS_sector"] = df_model["NAICS_str"].str[:2]
df_model.drop(columns=["NAICS_str"], errors="ignore", inplace=True)

# 7.7 Clean LowDoc and RevLineCr
def clean_yes_no(series):
    cleaned = series.astype(str).str.strip().str.upper()
    cleaned = cleaned.replace({"Y": "Yes", "YES": "Yes", "1": "Yes", "N": "No", "NO": "No", "0": "No", "NAN": "Unknown", "": "Unknown"})
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

# Replace inf/-inf with NaN
float_cols = df_model.select_dtypes(include=[np.floating]).columns
df_model[float_cols] = df_model[float_cols].replace([np.inf, -np.inf], np.nan)

print("df_model shape:", df_model.shape, "columns:", len(df_model.columns))


#%% 8. Register predictors — feature set v1 (23 numeric + 7 categorical)

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

print(f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)


#%% 9. Build X, y, and amount

X = df_model[numeric_cols + categorical_cols].copy()
y = df_model["y"].copy()
amount = df_model["DisbursementGross"].copy()

for col in categorical_cols:
    X[col] = X[col].fillna("Unknown").astype(str)

print("X shape:", X.shape, "default rate:", y.mean(), "amount missing:", amount.isna().sum())


#%% 10. Train / validation / test split

X_train_valid, X_test, y_train_valid, y_test, amount_train_valid, amount_test = train_test_split(
    X, y, amount, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

X_train, X_valid, y_train, y_valid, amount_train, amount_valid = train_test_split(
    X_train_valid, y_train_valid, amount_train_valid,
    test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_valid)

print("Train:", X_train.shape, "Valid:", X_valid.shape, "Test:", X_test.shape)
print("Train default:", y_train.mean(), "Valid default:", y_valid.mean())


#%% 11. Preprocessing objects

numeric_transformer_scaled = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

numeric_transformer_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
])

preprocess_scaled = ColumnTransformer([
    ("num", numeric_transformer_scaled, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

preprocess_tree = ColumnTransformer([
    ("num", numeric_transformer_tree, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# HGB needs dense input
preprocess_hgb = Pipeline(steps=[
    ("preprocess", preprocess_tree),
    ("to_dense", FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else X, accept_sparse=True)),
])

print("Preprocessing ready: preprocess_scaled, preprocess_tree, preprocess_hgb")


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
    TP = ((y_true_arr == 1) & (pred_default == 1)).sum()
    FN = ((y_true_arr == 1) & (pred_default == 0)).sum()
    FP = ((y_true_arr == 0) & (pred_default == 1)).sum()
    TN = ((y_true_arr == 0) & (pred_default == 0)).sum()
    profit = loan_profit_vector(y_true_arr, decision_approve, amount_series)
    return {
        "model": model_name, "threshold_default": threshold, "threshold_success": 1 - threshold,
        "accuracy": accuracy_score(y_true_arr, pred_default),
        "recall_default_sensitivity": recall_score(y_true_arr, pred_default, zero_division=0),
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
print("Approve-all profit:", approve_all_profit)
print("Deny-all profit:", deny_all_profit)


#%% 14A. KNN workspace — Owner: Hai An

if RUN_KNN:
    knn_best = KNeighborsClassifier(n_neighbors=51, weights="distance", p=2, n_jobs=-1)
    knn_pipe, knn_prob, knn_theory, knn_tuned, knn_threshold_table = fit_predict_evaluate(
        "KNN_k51_distance", knn_best, preprocess_obj=preprocess_scaled)

    print("\nKNN validation results:")
    print(pd.DataFrame([knn_theory, knn_tuned])[
        ["model","threshold_default","threshold_type","auc","brier","net_profit","approval_rate","runtime_seconds"]])

    print("\nTop 10 KNN thresholds:")
    print(knn_threshold_table.head(10)[["threshold_default","net_profit","approval_rate","approved_default_rate","auc"]])

# Hai An KNN: k=51, distance → profit ~$53.5M, AUC 0.861 (older features)
# With v1 features, expected improvement.


#%% 14B. Decision tree / bagging / RF / AdaBoost / HGB — Owner: Hai An

if RUN_TREE_MODELS:
    tree_results = []

    # --------- Random Forest ---------
    print("\n" + "=" * 60)
    print("Random Forest")
    print("=" * 60)

    rf_configs = [
        {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 25, "max_features": None, "label": "RF_n100_d16_l25"},
        {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 50, "max_features": None, "label": "RF_n100_d16_l50"},
        {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 50, "max_features": None, "label": "RF_n200_d16_l50"},
    ]
    for cfg in rf_configs:
        rf = RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                    min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
                                    n_jobs=-1, random_state=RANDOM_STATE)
        _, _, _, tuned, _ = fit_predict_evaluate(cfg["label"], rf, preprocess_obj=preprocess_tree)
        tree_results.append(tuned)
        print(f"  {cfg['label']}: Profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}, Thr={tuned['threshold_default']:.3f}")

    # --------- Bagging ---------
    print("\n" + "=" * 60)
    print("Bagging")
    print("=" * 60)
    base_bag = DecisionTreeClassifier(max_depth=16, min_samples_leaf=50, random_state=RANDOM_STATE)
    try:
        bag = BaggingClassifier(estimator=base_bag, n_estimators=100, max_samples=0.7,
                                bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE)
    except TypeError:
        bag = BaggingClassifier(base_estimator=base_bag, n_estimators=100, max_samples=0.7,
                                bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE)
    _, _, _, bag_tuned, _ = fit_predict_evaluate("Bagging_n100_d16_l50", bag, preprocess_obj=preprocess_tree)
    tree_results.append(bag_tuned)
    print(f"  Profit=${bag_tuned['net_profit']:,.0f}, AUC={bag_tuned['auc']:.4f}")

    # --------- AdaBoost ---------
    print("\n" + "=" * 60)
    print("AdaBoost")
    print("=" * 60)
    base_ada = DecisionTreeClassifier(max_depth=2, min_samples_leaf=200, random_state=RANDOM_STATE)
    try:
        boost = AdaBoostClassifier(estimator=base_ada, n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE)
    except TypeError:
        boost = AdaBoostClassifier(base_estimator=base_ada, n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE)
    _, _, _, boost_tuned, _ = fit_predict_evaluate("AdaBoost_d2_l200_n100", boost, preprocess_obj=preprocess_tree)
    tree_results.append(boost_tuned)
    print(f"  Profit=${boost_tuned['net_profit']:,.0f}, AUC={boost_tuned['auc']:.4f}")

    # --------- HGB — focused tuning around official best ---------
    if _HAS_HIST_GB:
        print("\n" + "=" * 60)
        print("HistGradientBoosting — focused tuning")
        print("=" * 60)

        hgb_configs = [
            {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 50,
             "l2_regularization": 0.1, "label": "HGB_i100_lr0.05_ln31"},
            {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 50,
             "l2_regularization": 0.1, "label": "HGB_i200_lr0.05_ln31"},
            {"max_iter": 200, "learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 50,
             "l2_regularization": 0.1, "label": "HGB_i200_lr0.03_ln31"},
            {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 100,
             "l2_regularization": 0.1, "label": "HGB_i200_lr0.05_ln15"},
            {"max_iter": 250, "learning_rate": 0.08, "max_leaf_nodes": 63, "min_samples_leaf": 50,
             "l2_regularization": 0.1, "label": "HGB_i250_lr0.08_ln63"},
        ]

        for cfg in hgb_configs:
            hgb = HistGradientBoostingClassifier(
                max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
                max_leaf_nodes=cfg["max_leaf_nodes"], min_samples_leaf=cfg["min_samples_leaf"],
                l2_regularization=cfg["l2_regularization"], random_state=RANDOM_STATE)
            _, _, _, tuned, _ = fit_predict_evaluate(cfg["label"], hgb, preprocess_obj=preprocess_hgb)
            tree_results.append(tuned)
            print(f"  {cfg['label']}: Profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}, Thr={tuned['threshold_default']:.3f}")

    # --------- Summary ---------
    print("\n" + "=" * 60)
    print("Tree-family summary (sorted by profit)")
    print("=" * 60)
    tree_df = pd.DataFrame(tree_results).sort_values("net_profit", ascending=False).reset_index(drop=True)
    cols = ["model", "threshold_default", "auc", "brier", "net_profit", "approval_rate", "approved_default_rate", "runtime_seconds"]
    print(tree_df[[c for c in cols if c in tree_df.columns]].head(15).to_string(index=False))

# Hai An tree-family notes (earlier run with fewer features):
# RF best: n100/d16/l25 → $68.9M, AUC 0.965
# HGB best: i250/lr0.08/ln63 → $68.3M, AUC 0.961
# With FULL v1 features, expect HGB to reach ~$70M.


#%% 16. Build validation leaderboard

model_results = []
var_names = ["knn_tuned", "rf_tuned", "bag_tuned", "boost_tuned"]
for vname in var_names:
    if vname in globals():
        model_results.append(globals()[vname])

# Add HGB results if available
if "tree_results" in globals():
    model_results.extend(tree_results)

baseline_rows = [
    {"model": "Approve All Baseline", "threshold_type": "baseline",
     "net_profit": approve_all_profit, "approval_rate": 1.0, "approved_default_rate": y_valid.mean()},
    {"model": "Deny All Baseline", "threshold_type": "baseline",
     "net_profit": deny_all_profit, "approval_rate": 0.0, "approved_default_rate": np.nan},
]

leaderboard = pd.DataFrame(baseline_rows + model_results)
leaderboard = leaderboard.sort_values("net_profit", ascending=False).reset_index(drop=True)

display_cols = ["model", "threshold_default", "auc", "brier", "net_profit",
                "approval_rate", "approved_default_rate", "runtime_seconds"]
print("\n=== Hai An Shared V1 Leaderboard ===")
print(leaderboard[[c for c in display_cols if c in leaderboard.columns]].to_string(index=False))


#%% 19. Final untouched test evaluation

if RUN_FINAL_TEST:
    pass
else:
    print("RUN_FINAL_TEST = False — skipping final test.")


#%% 20. Save outputs

print("Hai An Shared V1 Rerun complete.")
print("Set RUN_KNN = True and RUN_TREE_MODELS = True to run models.")
print("Do NOT run RUN_FINAL_TEST yet.")
