# -*- coding: utf-8 -*-
"""
SBA Final Project — End-to-End Workflow
Final test protected. EDA on development data. Full dev set for modeling.
Colab-compatible. Parquet + CSV fallback.
"""

#%% [markdown]
# # SBA Loan Default Project — Final Working Notebook
#
# Run cells one by one with Shift+Enter. Do not click "Run all" unless we agree.
# If a cell fails, stop and fix it before moving on.
#
# Protected sections: dataset loading, dev/test split, target, modeling data, train/valid split, profit function, threshold tuning, final test.
#
# You can edit: your EDA section, your feature block, your model block, your tuning search space.
#
# Target: y=0 paid in full, y=1 default. Profit rule: +5% if approve+paid, -25% if approve+default, 0 if deny.
#
# Final test is protected. I will run it after we freeze the final model and threshold.

#%% 1. Imports and Colab / Drive setup

import os, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

try: from sklearn.ensemble import HistGradientBoostingClassifier; _HAS_HGB = True
except ImportError: _HAS_HGB = False
try: import optuna; _HAS_OPTUNA = True
except ImportError: _HAS_OPTUNA = False
try: import xgboost as xgb; _HAS_XGB = True
except ImportError: _HAS_XGB = False
try: import lightgbm as lgb; _HAS_LGB = True
except ImportError: _HAS_LGB = False

pd.set_option("display.max_columns", None); pd.set_option("display.width", 200)
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive; drive.mount("/content/drive")

BASE_DIRS = [Path("."), Path("/content/drive/MyDrive/ML Final Project"), Path("/content/drive/MyDrive"),
             Path("/content/drive/Shareddrives/ML Final Project"), Path("/content/drive/Shared drives/ML Final Project"), Path("/content")]
CANDIDATES = ["research_outputs/sba_enriched_eda_dataset.parquet","sba_enriched_eda_dataset.parquet",
              "research_outputs/haianh_improved_eda_dataset.parquet","haianh_improved_eda_dataset.parquet"]
CSV_CANDIDATES = ["research_outputs/sba_enriched_eda_dataset.csv","sba_enriched_eda_dataset.csv"]

DATA_PATH = None; FILE_TYPE = None
for base in BASE_DIRS:
    for cand in CANDIDATES:
        p = base / cand
        if p.exists(): DATA_PATH = str(p); FILE_TYPE = "parquet"; break
    if DATA_PATH: break
if DATA_PATH is None:
    for base in BASE_DIRS:
        for cand in CSV_CANDIDATES:
            p = base / cand
            if p.exists(): DATA_PATH = str(p); FILE_TYPE = "csv"; break
        if DATA_PATH: break

print(f"Working dir: {os.getcwd()}")
print(f"Dataset: {DATA_PATH or 'NOT FOUND'} ({FILE_TYPE})")
if DATA_PATH is None:
    print("ERROR: Cannot find dataset. If using shared Drive, add a shortcut to My Drive and rerun.")
    sys.exit(1)

#%% 2. Load full finalized dataset

df_full = None
if FILE_TYPE == "parquet":
    try: df_full = pd.read_parquet(DATA_PATH)
    except Exception as e:
        print(f"Parquet failed: {e}")
        for base in BASE_DIRS:
            for cand in CSV_CANDIDATES:
                p = base / cand
                if p.exists(): DATA_PATH = str(p); FILE_TYPE = "csv"; break
            if FILE_TYPE == "csv": break
if FILE_TYPE == "csv" and df_full is None: df_full = pd.read_csv(DATA_PATH)
if df_full is None: print("ERROR: Could not load dataset."); sys.exit(1)

print(f"Loaded: {DATA_PATH} ({FILE_TYPE}), shape={df_full.shape}, DR={df_full['y'].mean():.4f}")

_PP = Path(DATA_PATH).resolve()
PROJECT_DIR = _PP.parents[1] if "research_outputs" in str(_PP) else _PP.parent
FINAL_OUT = PROJECT_DIR / "research_outputs" / "final_workflow_outputs"
FINAL_OUT.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {FINAL_OUT}")

#%% 3. Create era labels and NAICS string

df_full["era"] = "pre"
df_full.loc[df_full["approval_year"].between(2007, 2010), "era"] = "crisis"
df_full.loc[df_full["approval_year"] >= 2011, "era"] = "post"

df_full["NAICS_sector_str"] = (
    pd.to_numeric(df_full["NAICS_sector"], errors="coerce").astype("Int64").astype(str))

RANDOM_STATE = 1

#%% [markdown]
# # Data Split Design
#
# 1. `df_full` — full dataset
# 2. `df_dev` — development data for EDA, feature design, model training, and validation
# 3. `df_final_test` — untouched final test, do not use until the final model is frozen
#
# EDA runs on `df_dev`. Modeling uses the full development set. The dev set is split into train (fit) and validation (tune). Final test remains untouched.

#%% 4. Split full data into development and untouched final test

try:
    strat_key = df_full["y"].astype(str) + "_" + df_full["era"].astype(str)
    df_dev, df_final_test = train_test_split(df_full, test_size=0.20, random_state=RANDOM_STATE, stratify=strat_key)
except Exception:
    df_dev, df_final_test = train_test_split(df_full, test_size=0.20, random_state=RANDOM_STATE, stratify=df_full["y"])

print("Full:", len(df_full), f"DR={df_full['y'].mean():.4f}")
print("Dev:", len(df_dev), f"DR={df_dev['y'].mean():.4f}")
print("Test:", len(df_final_test), f"DR={df_final_test['y'].mean():.4f}")
for label, subset in [("Full",df_full),("Dev",df_dev),("Test",df_final_test)]:
    print(f"  {label} eras:", {e: len(subset[subset["era"]==e]) for e in ["pre","crisis","post"]})
print("Dev DR by era:", {e: round(df_dev[df_dev["era"]==e]["y"].mean(),4) for e in ["pre","crisis","post"]})

#%% 5. EDA on development data only

df_eda = df_dev.copy()
AMOUNT_COL = "DisbursementGross" if "DisbursementGross" in df_eda.columns else "GrAppv"
print("Amount column:", AMOUNT_COL)

OUT_DIR = FINAL_OUT / "baseline_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target distribution
tgt = df_eda["y"].value_counts().rename({0:"Paid",1:"Default"})
print("Target distribution:"); print(tgt)
pd.DataFrame({"outcome":tgt.index,"count":tgt.values}).to_csv(OUT_DIR/"target_distribution_table.csv",index=False)
fig,ax=plt.subplots(figsize=(5,4))
ax.bar(["Paid","Default"],tgt.values,color=["#2ecc71","#e74c3c"],edgecolor="black")
for i,v in enumerate(tgt.values): ax.text(i,v+len(df_eda)*0.005,f"{v:,}",ha="center",fontweight="bold")
ax.set_title("Loan Outcome Distribution"); plt.tight_layout(); plt.savefig(OUT_DIR/"target_distribution.png",dpi=120); plt.close()
print(f"Default rate: {df_eda['y'].mean():.4f}")

#%% [markdown]
# # Baseline EDA: Original SBA Risk Patterns
#
# Run this section to understand what the original SBA variables tell us before external data.
#
# Hai Anh: target variable, term bucket, loan-size bucket. Collect charts/tables. Write: which group is highest/lowest risk? Is the pattern monotonic? What business reason explains it?
#
# Hai An: employees, new vs existing, same-state lender. Collect charts/tables. Write: which borrower is riskier? Does the pattern hold across eras? What does this say about borrower fragility or lender information?
#
# Huyen Anh: NAICS sector, state, LowDoc, RevLineCr, UrbanRural, long-maturity indicator. Collect charts/tables. Write: which industries/states are highest risk? Which program variables look risky? Note that `RealEstate` means long-maturity indicator (Term >= 240), NOT the real estate industry (NAICS 53).

#%% 6. Descriptive statistics

desc_cols = ["Term","NoEmp","CreateJob","RetainedJob","GrAppv","SBA_Appv"]
desc_cols = [c for c in desc_cols if c in df_eda.columns]
desc_numeric = []
for col in desc_cols:
    s = df_eda[col].dropna()
    desc_numeric.append({"variable":col,"p5":s.quantile(0.05),"p25":s.quantile(0.25),
                         "median":s.median(),"p75":s.quantile(0.75),"p95":s.quantile(0.95),
                         "mean":s.mean(),"observations":len(s)})
pd.DataFrame(desc_numeric).round(2).to_csv(FINAL_OUT/"descriptive_numeric.csv",index=False)

cat_desc = ["State","BankState","NewExist","UrbanRural","NAICS_sector","LowDoc_clean","RevLineCr_clean"]
cat_desc = [c for c in cat_desc if c in df_eda.columns]
desc_cat = []
for col in cat_desc:
    for val in df_eda[col].dropna().unique():
        seg = df_eda[df_eda[col]==val]
        desc_cat.append({"variable":col,"level":str(val),"observations":len(seg),"default_rate":round(seg["y"].mean(),4)})
pd.DataFrame(desc_cat).to_csv(FINAL_OUT/"descriptive_categorical.csv",index=False)
print("Descriptive stats saved.")

#%% 7. Baseline EDA — original SBA variables

# Term
df_eda["_tb"] = pd.cut(df_eda["Term"], bins=[-1,60,180,np.inf], labels=["<60mo","60-180mo",">180mo"])
tbt = df_eda.groupby("_tb",observed=False)["y"].agg(["count","mean"]).round(4)
tbt.to_csv(OUT_DIR/"term_table.csv")
print("Term:"); print(tbt)
fig,ax=plt.subplots(figsize=(7,4))
ax.bar(tbt.index.astype(str),tbt["mean"]*100,color=["#e74c3c","#f39c12","#2ecc71"],edgecolor="black")
for i,(_,r) in enumerate(tbt.iterrows()): ax.text(i,r["mean"]*100+1,f'{r["mean"]*100:.1f}%',ha="center")
ax.set_title("Default Rate by Term"); ax.set_ylabel("Default Rate (%)")
plt.tight_layout(); plt.savefig(OUT_DIR/"term.png",dpi=120); plt.close()

# Loan size
df_eda["_sb"] = pd.cut(df_eda["GrAppv"],bins=[-1,50000,200000,np.inf],labels=["<$50K","$50-200K",">=$200K"])
sbt = df_eda.groupby("_sb",observed=False)["y"].agg(["count","mean"]).round(4)
sbt.to_csv(OUT_DIR/"loan_size_table.csv")
print("\nLoan size:"); print(sbt)
fig,ax=plt.subplots(figsize=(7,4))
ax.bar(sbt.index.astype(str),sbt["mean"]*100,color=["#e74c3c","#f39c12","#2ecc71"],edgecolor="black")
for i,(_,r) in enumerate(sbt.iterrows()): ax.text(i,r["mean"]*100+0.5,f'{r["mean"]*100:.1f}%',ha="center")
ax.set_title("Default Rate by Loan Size"); plt.tight_layout(); plt.savefig(OUT_DIR/"loan_size.png",dpi=120); plt.close()

# Employees
df_eda["_eb"] = pd.cut(df_eda["NoEmp"],bins=[-1,1,5,20,np.inf],labels=["0-1","2-5","6-20","21+"])
ebt = df_eda.groupby("_eb",observed=False)["y"].agg(["count","mean"]).round(4)
ebt.to_csv(OUT_DIR/"employee_table.csv")
print("\nFirm size:"); print(ebt)
fig,ax=plt.subplots(figsize=(7,4))
ax.bar(ebt.index.astype(str),ebt["mean"]*100,color=["#e74c3c","#e67e22","#f39c12","#2ecc71"],edgecolor="black")
for i,(_,r) in enumerate(ebt.iterrows()): ax.text(i,r["mean"]*100+0.3,f'{r["mean"]*100:.1f}%',ha="center")
ax.set_title("Default Rate by Number of Employees"); ax.set_ylabel("Default Rate (%)")
plt.tight_layout(); plt.savefig(OUT_DIR/"employee.png",dpi=120); plt.close()

# NewExist
ne = df_eda["NewExist"].map({1.0:"Existing",2.0:"New",0.0:"Unknown"})
net = df_eda.groupby(ne)["y"].agg(["count","mean"]).round(4)
net.to_csv(OUT_DIR/"newexist_table.csv")
print("\nNewExist:"); print(net)
fig,ax=plt.subplots(figsize=(5,4))
ax.bar(net.index.astype(str),net["mean"]*100,color=["#3498db","#e74c3c","#95a5a6"],edgecolor="black")
for i,(_,r) in enumerate(net.iterrows()): ax.text(i,r["mean"]*100+0.2,f'{r["mean"]*100:.1f}%',ha="center")
ax.set_title("Default Rate by Business Type"); ax.set_ylabel("Default Rate (%)")
plt.tight_layout(); plt.savefig(OUT_DIR/"newexist.png",dpi=120); plt.close()

# same_state_bank
sst = df_eda.groupby("same_state_bank")["y"].agg(["count","mean"]).round(4)
sst.to_csv(OUT_DIR/"same_state_bank_table.csv")
print("\nSame-state bank:"); print(sst)
fig,ax=plt.subplots(figsize=(5,4))
ax.bar(["Out-of-state","Same-state"],sst["mean"]*100,color=["#e74c3c","#2ecc71"],edgecolor="black")
for i,(_,r) in enumerate(sst.iterrows()): ax.text(i,r["mean"]*100+0.3,f'{r["mean"]*100:.1f}%',ha="center")
ax.set_title("Default Rate by Lender Location"); ax.set_ylabel("Default Rate (%)")
plt.tight_layout(); plt.savefig(OUT_DIR/"same_state_bank.png",dpi=120); plt.close()

# State
st = df_eda.groupby("State")["y"].agg(["count","mean"]).query("count>=100").sort_values("mean",ascending=False).round(4)
st.to_csv(OUT_DIR/"top_states_table.csv")
print("\nTop states:"); print(st.head(15))
fig,ax=plt.subplots(figsize=(9,5))
top15=st.head(15); ax.barh(top15.index,top15["mean"]*100,color="#e74c3c",edgecolor="black"); ax.invert_yaxis()
ax.set_title("Top 15 States by Default Rate"); plt.tight_layout(); plt.savefig(OUT_DIR/"state.png",dpi=120); plt.close()

# NAICS
ns = df_eda.groupby("NAICS_sector")["y"].agg(["count","mean"]).query("count>=50").sort_values("mean",ascending=False).round(4)
ns.to_csv(OUT_DIR/"naics_table.csv")
print("\nNAICS:"); print(ns.head(10))
fig,ax=plt.subplots(figsize=(8,5))
top_ns=ns.head(10); ax.barh(top_ns.index,top_ns["mean"]*100,color="#e74c3c",edgecolor="black"); ax.invert_yaxis()
ax.set_title("Default Rate by NAICS Sector"); ax.set_xlabel("Default Rate (%)")
plt.tight_layout(); plt.savefig(OUT_DIR/"naics.png",dpi=120); plt.close()

# UrbanRural, LowDoc, RevLineCr, RealEstate
for col, fname in [("UrbanRural","urbanrural"),("LowDoc_clean","lowdoc"),("RevLineCr_clean","revlinecr")]:
    t = df_eda.groupby(col)["y"].agg(["count","mean"]).round(4)
    t.to_csv(OUT_DIR/f"{fname}_table.csv")
    print(f"\n{col}:"); print(t)

ret = df_eda.groupby("RealEstate")["y"].agg(["count","mean"]).round(4)
ret.to_csv(OUT_DIR/"long_maturity_table.csv")
print("\nLong-maturity (RealEstate = Term>=240):"); print(ret)
print("NOTE: RealEstate = long-maturity indicator, NOT real estate industry (NAICS 53)")
fig,ax=plt.subplots(figsize=(5,4))
ax.bar(["Term < 240","Term >= 240"],ret["mean"]*100,color=["#e74c3c","#2ecc71"],edgecolor="black")
for i,(_,r) in enumerate(ret.iterrows()): ax.text(i,r["mean"]*100+0.3,f'{r["mean"]*100:.1f}%',ha="center")
ax.set_title("Default Rate by Long-Maturity Indicator"); ax.set_ylabel("Default Rate (%)")
plt.tight_layout(); plt.savefig(OUT_DIR/"long_maturity.png",dpi=120); plt.close()

# Era-split
for feat, bins, labels in [("Term",[-1,60,180,np.inf],["<60mo","60-180mo",">180mo"]),
                            ("GrAppv",[-1,50000,200000,np.inf],["<$50K","$50-200K",">=$200K"])]:
    df_eda["_xx"] = pd.cut(df_eda[feat],bins=bins,labels=labels)
    for era in ["pre","crisis","post"]:
        s=df_eda[df_eda["era"]==era]
        for lb in labels:
            seg=s[s["_xx"]==lb]
            if len(seg)>30: print(f"  {feat} {era} {lb}: DR={seg['y'].mean():.4f} (n={len(seg):,})")

df_eda.drop(columns=["_tb","_sb","_eb","_xx"],errors="ignore",inplace=True)
print("\nBaseline EDA complete.")

#%% [markdown]
# # External EDA — Hai An
#
# You cover: real estate industry x HPI, construction x HPI, state x industry hotspots, supply chain x inflation.
#
# Collect: HPI tables for NAICS 53 and 23, state x industry hotspot table, inflation x supply-chain table.
# Write: what pattern did you find? Keep, reject, or report only? What does it mean for loan approval?

#%% 8. External EDA: Hai An — Industry / Business Environment

OUT_HA = FINAL_OUT / "hai_an_external_eda"; OUT_HA.mkdir(parents=True, exist_ok=True)

if "state_hpi_negative_flag" in df_eda.columns:
    df_eda["_is_re"] = (df_eda["NAICS_sector_str"]=="53").astype(int)
    h2 = df_eda.groupby(["_is_re","state_hpi_negative_flag"])["y"].agg(["count","mean"]).round(4)
    h2.to_csv(OUT_HA/"real_estate_hpi_table.csv")
    print("Real estate x HPI:"); print(h2)

    df_eda["_is_const"] = (df_eda["NAICS_sector_str"]=="23").astype(int)
    h3 = df_eda.groupby(["_is_const","state_hpi_negative_flag"])["y"].agg(["count","mean"]).round(4)
    h3.to_csv(OUT_HA/"construction_hpi_table.csv")
    print("Construction x HPI:"); print(h3)

    ch = df_eda[(df_eda["_is_const"]==1)&(df_eda["state_hpi_negative_flag"]==1)]
    if len(ch)>50:
        chs = ch.groupby("State")["y"].agg(["count","mean"]).query("count>=20").sort_values("mean",ascending=False).round(4)
        chs.to_csv(OUT_HA/"construction_hpi_state_hotspots.csv")

    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(["No HPI/Not const","HPI/Not const","No HPI/Const","HPI+Const"],
           [df_eda[(df_eda["_is_const"]==0)&(df_eda["state_hpi_negative_flag"]==0)]["y"].mean()*100,
            df_eda[(df_eda["_is_const"]==0)&(df_eda["state_hpi_negative_flag"]==1)]["y"].mean()*100,
            df_eda[(df_eda["_is_const"]==1)&(df_eda["state_hpi_negative_flag"]==0)]["y"].mean()*100,
            df_eda[(df_eda["_is_const"]==1)&(df_eda["state_hpi_negative_flag"]==1)]["y"].mean()*100],
           color=["#2ecc71","#f39c12","#e67e22","#e74c3c"],edgecolor="black")
    ax.set_title("Construction x HPI Negative"); plt.xticks(rotation=15,ha="right")
    plt.tight_layout(); plt.savefig(OUT_HA/"construction_hpi.png",dpi=120); plt.close()

if "high_inflation_flag_lag1" in df_eda.columns:
    supply_naics = ["23","31","32","33","42","48","49"]
    df_eda["_supply"] = df_eda["NAICS_sector_str"].isin(supply_naics).astype(int)
    sc = df_eda.groupby(["_supply","high_inflation_flag_lag1"])["y"].agg(["count","mean"]).round(4)
    sc.to_csv(OUT_HA/"supply_chain_inflation_table.csv")
    print("Supply chain x inflation:"); print(sc)

df_eda.drop(columns=["_is_re","_is_const","_supply"],errors="ignore",inplace=True)
print("Hai An external EDA done.")

#%% [markdown]
# # External EDA — Hai Anh
#
# You cover: inflation, oil shock, income decline, bank failures, local economy variables.
#
# Collect: high inflation table, oil shock table, income decline table, bank failure table, local economy summary.
# Write: what pattern did you find? Keep, reject, or report only? What does it mean for loan approval?

#%% 9. External EDA: Hai Anh — Local Economy / Weak Borrowers

OUT_HAH = FINAL_OUT / "hai_anh_external_eda"; OUT_HAH.mkdir(parents=True, exist_ok=True)

if "high_inflation_flag_lag1" in df_eda.columns:
    hi = df_eda.groupby("high_inflation_flag_lag1")["y"].agg(["count","mean"]).round(4)
    hi.to_csv(OUT_HAH/"high_inflation_table.csv")
    print("High inflation:"); print(hi)

    df_eda["_eb2"] = pd.cut(df_eda["NoEmp"],bins=[-1,5,20,np.inf],labels=["Micro(0-5)","Small(6-20)","Large(21+)"])
    hi_firm = df_eda.groupby(["_eb2","high_inflation_flag_lag1"],observed=False)["y"].agg(["count","mean"]).round(4)
    hi_firm.to_csv(OUT_HAH/"inflation_firm_size_table.csv")
    print("Inflation x firm size:"); print(hi_firm)

    fig,ax=plt.subplots(figsize=(8,5))
    sizes=["Micro(0-5)","Small(6-20)","Large(21+)"]
    x=np.arange(len(sizes)); w=0.3
    for j,(lbl,v) in enumerate([("Normal",0),("High infl",1)]):
        vals=[df_eda[(df_eda["_eb2"]==s)&(df_eda["high_inflation_flag_lag1"]==v)]["y"].mean()*100 for s in sizes]
        ax.bar(x+j*w,vals,w,label=lbl,color=["#2ecc71","#e74c3c"][j],edgecolor="black")
    ax.set_xticks(x+w/2); ax.set_xticklabels(sizes)
    ax.set_title("Inflation x Firm Size"); ax.set_ylabel("Default Rate (%)"); ax.legend()
    plt.tight_layout(); plt.savefig(OUT_HAH/"inflation_firm_size.png",dpi=120); plt.close()

if "oil_shock_flag_lag1" in df_eda.columns:
    oi = df_eda.groupby("oil_shock_flag_lag1")["y"].agg(["count","mean"]).round(4)
    oi.to_csv(OUT_HAH/"oil_shock_table.csv")
    print("Oil shock:"); print(oi)

if "real_income_decline_flag_lag1" in df_eda.columns:
    ri = df_eda.groupby("real_income_decline_flag_lag1")["y"].agg(["count","mean"]).round(4)
    ri.to_csv(OUT_HAH/"income_decline_table.csv")
    print("Income decline:"); print(ri)

if "state_bank_failures_12m" in df_eda.columns and "state_bank_failures_data_available" in df_eda.columns:
    fdic_ok = df_eda[df_eda["state_bank_failures_data_available"]==1]
    bk = fdic_ok.groupby((fdic_ok["state_bank_failures_12m"]>0).astype(int))["y"].agg(["count","mean"]).round(4)
    bk.to_csv(OUT_HAH/"bank_failures_table.csv")
    print("Bank failures:"); print(bk)

if all(c in df_eda.columns for c in ["high_inflation_flag_lag1","oil_shock_flag_lag1"]):
    df_eda["_stress"] = df_eda["high_inflation_flag_lag1"].fillna(0).astype(int)+df_eda["oil_shock_flag_lag1"].fillna(0).astype(int)
    ss = df_eda.groupby("_stress")["y"].agg(["count","mean"]).round(4)
    ss.to_csv(OUT_HAH/"stress_score_table.csv")
    print("Stress score:"); print(ss)
    fig,ax=plt.subplots(figsize=(6,4))
    vals=df_eda.groupby("_stress")["y"].mean()*100
    ax.bar(vals.index.astype(int).astype(str),vals.values,color=["#2ecc71","#f39c12","#e74c3c"],edgecolor="black")
    ax.set_title("Stress Score vs Default Rate"); ax.set_ylabel("Default Rate (%)"); ax.set_xlabel("Flags")
    plt.tight_layout(); plt.savefig(OUT_HAH/"stress_score.png",dpi=120); plt.close()

df_eda.drop(columns=["_eb2","_stress"],errors="ignore",inplace=True)
print("Hai Anh external EDA done.")

#%% [markdown]
# # External EDA — Huyen Anh
#
# You cover: FEMA disaster exposure, hurricane/flood/fire/severe storm, QCEW patterns.
#
# Collect: disaster-type table, FEMA lookback table, QCEW table, era-split table.
# Write: what pattern did you find? Keep, reject, or report only? What does it mean for loan approval?

#%% 10. External EDA: Huyen Anh — Disaster / Housing / External Shocks

OUT_HUY = FINAL_OUT / "huyen_anh_external_eda"; OUT_HUY.mkdir(parents=True, exist_ok=True)

if "NAICS_sector_str" not in df_eda.columns:
    df_eda["NAICS_sector_str"] = pd.to_numeric(df_eda["NAICS_sector"],errors="coerce").astype("Int64").astype(str)

print("FEMA columns:", [c for c in df_eda.columns if c.startswith("fema_")])
print("QCEW columns:", [c for c in df_eda.columns if c.startswith("qcew_")])

fema_12m = [c for c in df_eda.columns if c.startswith("fema_") and c.endswith("_12m")]
fema_rows = []
for col in fema_12m:
    e0=df_eda[df_eda[col]==0]; e1=df_eda[df_eda[col]==1]
    if len(e1)<20: continue
    fema_rows.append({"variable":col,"unexposed_n":len(e0),"unexposed_dr":e0["y"].mean(),
                       "exposed_n":len(e1),"exposed_dr":e1["y"].mean(),
                       "delta_pp":round((e1["y"].mean()-e0["y"].mean())*100,2)})
if fema_rows:
    fema_df = pd.DataFrame(fema_rows).sort_values("delta_pp",ascending=False)
    fema_df.to_csv(OUT_HUY/"fema_disaster_type_default_rates.csv",index=False)
    print("\nFEMA disaster types:"); print(fema_df.to_string(index=False))
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(fema_df)); w=0.35
    ax.bar(x-w/2,fema_df["unexposed_dr"]*100,w,label="Unexposed",color="#2ecc71",edgecolor="black")
    ax.bar(x+w/2,fema_df["exposed_dr"]*100,w,label="Exposed",color="#e74c3c",edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("fema_","").replace("_12m","") for c in fema_df["variable"]],rotation=45,ha="right")
    ax.set_title("Default Rate by FEMA Disaster Type (12m)"); ax.set_ylabel("Default Rate (%)"); ax.legend()
    plt.tight_layout(); plt.savefig(OUT_HUY/"fema_disaster_type_default_rates.png",dpi=120); plt.close()
else:
    empty_fema = pd.DataFrame(columns=["variable","unexposed_n","unexposed_dr","exposed_n","exposed_dr","delta_pp"])
    empty_fema.to_csv(OUT_HUY/"fema_disaster_type_default_rates.csv",index=False)
    print("\nFEMA disaster types: no type has enough exposed observations.")

# Tourism
df_eda["tourism_industry_flag"] = df_eda["NAICS_sector_str"].isin(["44","45","71","72"]).astype(int)
tr = df_eda.groupby("tourism_industry_flag")["y"].agg(n="count",dr="mean").round(4)
tr.to_csv(OUT_HUY/"tourism_industry_default_rate.csv"); print("Tourism:"); print(tr)

# Coastal
coastal = ["AL","AK","CA","CT","DE","FL","GA","HI","LA","ME","MD","MA","MS","NH","NJ","NY","NC","OR","RI","SC","TX","VA","WA"]
df_eda["coastal_state_flag"] = df_eda["State"].isin(coastal).astype(int)
cs = df_eda.groupby("coastal_state_flag")["y"].agg(n="count",dr="mean").round(4)
cs.to_csv(OUT_HUY/"coastal_state_default_rates.csv")

# Era-split disasters
era_split_cols = [c for c in ["fema_hurricane_12m","fema_fire_12m","fema_flood_12m","fema_severe_storm_12m","state_hpi_negative_flag"] if c in df_eda.columns]
era_rows = []
for col in era_split_cols:
    for e in ["pre","crisis","post"]:
        s=df_eda[df_eda["era"]==e]
        for v in [0,1]:
            seg=s[s[col]==v]
            if len(seg)>30: era_rows.append({"variable":col,"era":e,"exposed":v,"n":len(seg),"dr":seg["y"].mean()})
if era_rows: pd.DataFrame(era_rows).to_csv(OUT_HUY/"era_split_disaster_default_rates.csv",index=False)

# Hurricane by NAICS
if "fema_hurricane_12m" in df_eda.columns:
    h_naics = df_eda.groupby("NAICS_sector_str").apply(lambda g: pd.Series({
        "total_n":len(g),"hurr_n":(g["fema_hurricane_12m"]==1).sum(),
        "no_hurr_dr":g[g["fema_hurricane_12m"]==0]["y"].mean(),
        "hurr_dr":g[g["fema_hurricane_12m"]==1]["y"].mean() if (g["fema_hurricane_12m"]==1).sum()>=20 else np.nan})).reset_index()
    h_naics["delta_pp"] = ((h_naics["hurr_dr"]-h_naics["no_hurr_dr"])*100).round(2)
    h_naics.dropna(subset=["hurr_dr"]).sort_values("delta_pp",ascending=False).to_csv(OUT_HUY/"hurricane_by_naics_delta.csv",index=False)

# Fire by state
if "fema_fire_12m" in df_eda.columns:
    f_state = df_eda.groupby("State").apply(lambda g: pd.Series({
        "total_n":len(g),"fire_n":(g["fema_fire_12m"]==1).sum(),
        "no_fire_dr":g[g["fema_fire_12m"]==0]["y"].mean(),
        "fire_dr":g[g["fema_fire_12m"]==1]["y"].mean() if (g["fema_fire_12m"]==1).sum()>=20 else np.nan})).reset_index()
    f_state["delta_pp"] = ((f_state["fire_dr"]-f_state["no_fire_dr"])*100).round(2)
    f_state.dropna(subset=["fire_dr"]).sort_values("delta_pp",ascending=False).to_csv(OUT_HUY/"fire_by_state_delta.csv",index=False)

# QCEW audit
qcew_cols = [c for c in df_eda.columns if c.startswith("qcew_")]
if qcew_cols:
    qcew_audit = []
    for col in qcew_cols:
        s=df_eda[col].dropna()
        if len(s)>100: qcew_audit.append({"column":col,"count":len(s),"mean":s.mean(),"p5":s.quantile(0.05),"median":s.median(),"p95":s.quantile(0.95)})
    pd.DataFrame(qcew_audit).to_csv(OUT_HUY/"qcew_column_summary.csv",index=False)

print(f"\nHuyen Anh external EDA complete. Outputs: {OUT_HUY}")

#%% 11. Modeling data — full development set

df_model = df_dev.copy()

print(f"Modeling data: {len(df_model):,}, DR={df_model['y'].mean():.4f}")
for e in ["pre", "crisis", "post"]:
    era_slice = df_model[df_model["era"] == e]
    print(f"  {e}: {len(era_slice):,}, DR={era_slice['y'].mean():.4f}")

# df_model: full development set used for train/validation. FEATURE_MODE: "sba_only" or "sba_plus_external".

#%% [markdown]
# # Feature Engineering Sandbox
#
# Add model features from your EDA here. Rules:
#
# 1. Create columns on `df_model`. 2. Append names to `ADDED_NUMERIC_FEATURES` or `ADDED_CATEGORICAL_FEATURES`. 3. Add a row to `FEATURE_NOTES` with feature, your name, one-sentence business reason. 4. Do not modify y, df_final_test, the split, the profit function, or threshold tuning.
#
# Your feature is not report-ready unless you can explain it in one sentence.
#
# Important: if you create a new feature here, also recreate it inside `add_engineered_features_for_model(data)`. Otherwise the final test set will not have that feature.

#%% 12. Feature Engineering Sandbox

ADDED_NUMERIC_FEATURES = []
ADDED_CATEGORICAL_FEATURES = []
FEATURE_NOTES = []
EXPERIMENT_LOG = []

# ---- Hai An — industry and housing features ----
if "NAICS_sector_str" in df_model.columns and "state_hpi_negative_flag" in df_model.columns:
    is_re = df_model["NAICS_sector_str"] == "53"
    is_const = df_model["NAICS_sector_str"] == "23"
    hpi_down = df_model["state_hpi_negative_flag"] == 1

    df_model["real_estate_industry_hpi"] = (is_re & hpi_down).astype(int)
    df_model["construction_hpi"] = (is_const & hpi_down).astype(int)
    ADDED_NUMERIC_FEATURES.extend(["real_estate_industry_hpi", "construction_hpi"])
    FEATURE_NOTES.append({"feature":"real_estate_industry_hpi","owner":"Hai An","reason":"Real estate businesses may be riskier when local housing prices fall"})
    FEATURE_NOTES.append({"feature":"construction_hpi","owner":"Hai An","reason":"Construction firms may face demand drop when HPI is negative"})

if "high_inflation_flag_lag1" in df_model.columns and "NAICS_sector_str" in df_model.columns:
    supply_codes = ["23","31","32","33","42","48","49"]
    df_model["supply_chain_inflation"] = (df_model["NAICS_sector_str"].isin(supply_codes) & (df_model["high_inflation_flag_lag1"]==1)).astype(int)
    ADDED_NUMERIC_FEATURES.append("supply_chain_inflation")
    FEATURE_NOTES.append({"feature":"supply_chain_inflation","owner":"Hai An","reason":"Supply chain sectors may struggle during high inflation periods"})

# ---- Hai Anh features ----
if "high_inflation_flag_lag1" in df_model.columns and "NoEmp" in df_model.columns:
    df_model["inflation_micro_business"] = (
        (df_model["high_inflation_flag_lag1"] == 1) & (df_model["NoEmp"] <= 5)
    ).astype(int)
    ADDED_NUMERIC_FEATURES.append("inflation_micro_business")
    FEATURE_NOTES.append({"feature":"inflation_micro_business","owner":"Hai Anh","reason":"Small firms may be more exposed when inflation raises operating costs"})

if "oil_shock_flag_lag1" in df_model.columns and "NAICS_sector_str" in df_model.columns:
    supply_codes = ["23","31","32","33","42","48","49"]
    df_model["oil_shock_supply_chain"] = (
        (df_model["oil_shock_flag_lag1"] == 1) & df_model["NAICS_sector_str"].isin(supply_codes)
    ).astype(int)
    ADDED_NUMERIC_FEATURES.append("oil_shock_supply_chain")
    FEATURE_NOTES.append({"feature":"oil_shock_supply_chain","owner":"Hai Anh","reason":"Transport/construction/mfg/wholesale may be more sensitive to fuel cost shocks"})

if "real_income_decline_flag_lag1" in df_model.columns and "NAICS_sector_str" in df_model.columns:
    consumer_codes = ["44","45","71","72"]
    df_model["income_decline_consumer_sector"] = (
        (df_model["real_income_decline_flag_lag1"] == 1) & df_model["NAICS_sector_str"].isin(consumer_codes)
    ).astype(int)
    ADDED_NUMERIC_FEATURES.append("income_decline_consumer_sector")
    FEATURE_NOTES.append({"feature":"income_decline_consumer_sector","owner":"Hai Anh","reason":"Consumer-facing businesses may be more exposed when local real income falls"})

if "state_bank_failures_12m" in df_model.columns:
    df_model["recent_bank_failure_flag"] = (df_model["state_bank_failures_12m"].fillna(0) > 0).astype(int)
    ADDED_NUMERIC_FEATURES.append("recent_bank_failure_flag")
    FEATURE_NOTES.append({"feature":"recent_bank_failure_flag","owner":"Hai Anh","reason":"Recent bank failures may signal local credit-market stress"})

# ---- Huyen Anh features ----
tourism_codes = ["44","45","71","72"]
coastal_list = ["AL","AK","CA","CT","DE","FL","GA","HI","LA","ME","MD","MA","MS","NH","NJ","NY","NC","OR","RI","SC","TX","VA","WA"]
ds_codes = ["23","44","45","48","49","51","71","72"]

df_model["tourism_industry_flag"] = df_model["NAICS_sector_str"].isin(tourism_codes).astype(int)
df_model["coastal_state_flag"] = df_model["State"].isin(coastal_list).astype(int)
df_model["disaster_sensitive_sector_flag"] = df_model["NAICS_sector_str"].isin(ds_codes).astype(int)
ADDED_NUMERIC_FEATURES.extend(["tourism_industry_flag","coastal_state_flag","disaster_sensitive_sector_flag"])
FEATURE_NOTES.append({"feature":"tourism_industry_flag","owner":"Huyen Anh","reason":"Tourism industries (NAICS 44,45,71,72) are sensitive to travel disruptions"})
FEATURE_NOTES.append({"feature":"coastal_state_flag","owner":"Huyen Anh","reason":"Coastal states face recurring hurricane/flood exposure"})
FEATURE_NOTES.append({"feature":"disaster_sensitive_sector_flag","owner":"Huyen Anh","reason":"Some NAICS sectors are structurally more vulnerable to disasters"})

if "fema_hurricane_12m" in df_model.columns:
    df_model["tourism_hurricane"] = df_model["tourism_industry_flag"] * df_model["fema_hurricane_12m"]
    df_model["coastal_hurricane"] = df_model["coastal_state_flag"] * df_model["fema_hurricane_12m"]
    df_model["disaster_sensitive_hurricane"] = df_model["disaster_sensitive_sector_flag"] * df_model["fema_hurricane_12m"]
    ADDED_NUMERIC_FEATURES.extend(["tourism_hurricane","coastal_hurricane","disaster_sensitive_hurricane"])

pd.DataFrame(FEATURE_NOTES).to_csv(FINAL_OUT/"feature_notes.csv",index=False)
print("Added features:", ADDED_NUMERIC_FEATURES)

#%% 13. Feature setup

BASELINE_NUMERIC = ["Term","NoEmp","CreateJob","RetainedJob","GrAppv","SBA_Appv",
    "Portion","unguaranteed_ratio","unguaranteed_amount","jobs_total","jobs_per_dollar",
    "same_state_bank","RealEstate","approval_year","ApprovalFY_clean",
    "log_GrAppv","log_SBA_Appv","log_NoEmp"]
BASELINE_CATEGORICAL = ["State","BankState","NewExist","UrbanRural","NAICS_sector","LowDoc_clean","RevLineCr_clean"]
EXTERNAL_NUMERIC = ["state_hpi_negative_flag","state_hpi_growth_1y","state_hpi_drawdown_2y",
    "high_inflation_flag_lag1","cpi_yoy_change_lag1","oil_shock_flag_lag1","oil_yoy_change_lag1",
    "real_income_decline_flag_lag1","real_income_growth_1y_lag1",
    "fema_hurricane_12m","fema_fire_12m","fema_flood_12m","fema_severe_storm_12m",
    "state_bank_failures_12m","state_bank_failures_24m","industry_contraction_flag"]

baseline_avail = [c for c in BASELINE_NUMERIC if c in df_model.columns]
cat_avail = [c for c in BASELINE_CATEGORICAL if c in df_model.columns]
ext_avail = [c for c in EXTERNAL_NUMERIC if c in df_model.columns]
added_num_avail = [c for c in ADDED_NUMERIC_FEATURES if c in df_model.columns]
added_cat_avail = [c for c in ADDED_CATEGORICAL_FEATURES if c in df_model.columns]

FEATURE_MODE = "sba_plus_external"

if FEATURE_MODE == "sba_only":
    all_numeric = baseline_avail
    all_categorical = cat_avail
else:
    all_numeric = baseline_avail + ext_avail + added_num_avail
    all_categorical = cat_avail + added_cat_avail

print(f"FEATURE MODE: {FEATURE_MODE}")
print(f"Numeric: {len(all_numeric)}, Categorical: {len(all_categorical)}")

#%% 14. Build X, y, amount

X = df_model[all_numeric + all_categorical].copy()
y = df_model["y"].copy()
amount = df_model[AMOUNT_COL].copy()

for col in all_categorical:
    if col in X.columns: X[col] = X[col].fillna("Unknown").astype(str)

print("X shape:", X.shape, "DR:", y.mean())

#%% 15. Train / validation split (80/20)

try:
    model_strat_key = y.astype(str) + "_" + df_model["era"].astype(str)
    X_train, X_valid, y_train, y_valid, amount_train, amount_valid = train_test_split(
        X, y, amount, test_size=0.20, random_state=RANDOM_STATE, stratify=model_strat_key)
except Exception:
    X_train, X_valid, y_train, y_valid, amount_train, amount_valid = train_test_split(
        X, y, amount, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

print("Train:", X_train.shape, f"DR={y_train.mean():.4f}")
print("Valid:", X_valid.shape, f"DR={y_valid.mean():.4f}")
print("Final test (untouched):", len(df_final_test))

#%% 16. Preprocessing

try:
    _ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    ohe_kw = dict(handle_unknown="ignore", sparse_output=True)
    ohe_kw_dense = dict(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe_kw = dict(handle_unknown="ignore", sparse=True)
    ohe_kw_dense = dict(handle_unknown="ignore", sparse=False)

num_scaled = Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
num_tree = Pipeline([("imputer",SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(**ohe_kw))])
cat_pipe_dense = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(**ohe_kw_dense))])

preprocess_scaled = ColumnTransformer([("num",num_scaled,all_numeric),("cat",cat_pipe,all_categorical)])
preprocess_scaled_dense = ColumnTransformer([("num",num_scaled,all_numeric),("cat",cat_pipe_dense,all_categorical)])
preprocess_tree = ColumnTransformer([("num",num_tree,all_numeric),("cat",cat_pipe,all_categorical)])
preprocess_hgb = Pipeline([("preprocess",preprocess_tree),
    ("to_dense",FunctionTransformer(lambda X: X.toarray() if hasattr(X,"toarray") else X, accept_sparse=True))])
print("Preprocessing ready.")

#%% 17. Profit and evaluation functions

THEORETICAL_DEFAULT_THRESHOLD = 1/6

def loan_profit_vector(y_true, decision_approve, amount_series):
    a=np.asarray(amount_series,dtype=float); yt=np.asarray(y_true); ap=np.asarray(decision_approve).astype(bool)
    return np.where(ap, np.where(yt==0, 0.05*a, -0.25*a), 0.0)

def evaluate_prob_model(model_name, y_true, prob_default, amount_series, threshold):
    yt=np.asarray(y_true); pd=np.asarray(prob_default)
    pred=(pd>threshold).astype(int); dec=pd<=threshold
    TP=((yt==1)&(pred==1)).sum(); FN=((yt==1)&(pred==0)).sum()
    FP=((yt==0)&(pred==1)).sum(); TN=((yt==0)&(pred==0)).sum()
    profit=loan_profit_vector(yt,dec,amount_series)
    return {"model":model_name,"threshold_default":threshold,"threshold_success":1-threshold,
            "accuracy":accuracy_score(yt,pred),"recall_default_sensitivity":recall_score(yt,pred,zero_division=0),
            "specificity_paid":TN/(TN+FP)if(TN+FP)>0 else np.nan,"precision_default":precision_score(yt,pred,zero_division=0),
            "f1_default":f1_score(yt,pred,zero_division=0),"auc":roc_auc_score(yt,pd),"brier":brier_score_loss(yt,pd),
            "net_profit":profit.sum(),"approval_rate":dec.mean(),
            "approved_default_rate":yt[dec].mean()if dec.sum()>0 else np.nan,
            "denied_default_rate":yt[~dec].mean()if(~dec).sum()>0 else np.nan,
            "TP_default_denied":int(TP),"FN_default_approved":int(FN),"FP_paid_denied":int(FP),"TN_paid_approved":int(TN)}

def tune_threshold_by_profit(model_name, y_true, prob_default, amount_series, thresholds=None):
    if thresholds is None: thresholds=np.linspace(0.01,0.60,120)
    thresholds=np.unique(np.append(thresholds,THEORETICAL_DEFAULT_THRESHOLD))
    rows=[evaluate_prob_model(model_name,y_true,prob_default,amount_series,th) for th in thresholds]
    return pd.DataFrame(rows).sort_values("net_profit",ascending=False).reset_index(drop=True)

def fit_predict_evaluate(model_name, estimator, preprocess_obj, owner=""):
    start=time.perf_counter()
    pipe=Pipeline([("preprocess",preprocess_obj),("model",estimator)])
    pipe.fit(X_train,y_train); prob=pipe.predict_proba(X_valid)[:,1]; runtime=time.perf_counter()-start
    theory=evaluate_prob_model(model_name,y_valid,prob,amount_valid,THEORETICAL_DEFAULT_THRESHOLD)
    theory["threshold_type"]="theoretical_1_over_6"; theory["runtime_seconds"]=runtime; theory["owner"]=owner
    tt=tune_threshold_by_profit(model_name,y_valid,prob,amount_valid)
    tuned=tt.iloc[0].to_dict()
    tuned["threshold_type"]="validation_profit_tuned"; tuned["runtime_seconds"]=runtime
    tuned["owner"]=owner; tuned["feature_mode"]=FEATURE_MODE
    return pipe, prob, theory, tuned, tt

def append_and_save_results(result_path, threshold_path, result_rows, threshold_tables):
    existing = []
    if result_path.exists():
        existing.append(pd.read_csv(result_path))
    if result_rows:
        existing.append(pd.DataFrame(result_rows))
    if existing:
        out = pd.concat(existing, ignore_index=True)
        if "model" in out.columns:
            out = out.drop_duplicates(subset=["model","feature_mode"], keep="last")
        out.to_csv(result_path, index=False)
    existing_tt = []
    if threshold_path.exists():
        existing_tt.append(pd.read_csv(threshold_path))
    if threshold_tables:
        existing_tt.append(pd.concat(threshold_tables, ignore_index=True))
    if existing_tt:
        pd.concat(existing_tt, ignore_index=True).to_csv(threshold_path, index=False)

def make_xgb_classifier(**kwargs):
    """GPU-aware XGBoost, falls back to CPU if GPU not available."""
    base = dict(random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0)
    base.update(kwargs)
    try:
        return xgb.XGBClassifier(**base, tree_method="hist", device="cuda")
    except TypeError:
        return xgb.XGBClassifier(**base, tree_method="gpu_hist")

#%% 18. Baseline policies

approve_all_profit = loan_profit_vector(y_valid,np.ones(len(y_valid),dtype=bool),amount_valid).sum()
deny_all_profit = 0.0
print("Approve-all profit:", approve_all_profit)
print("Deny-all profit:", deny_all_profit)

#%% [markdown]
# # Advanced Model Packages
#
# We use Optuna, XGBoost, and LightGBM. If a package is missing, run: `!pip install optuna xgboost lightgbm` and rerun the import cell.
#
# Run your core models first. Then run the advanced/tuned block. Start with 20 trials. Increase if runtime is acceptable. Save the result and compare with your core model. Your model section is not done until you try at least one stronger tuning/advanced model.

#%% [markdown]
# # Model Section — Hai Anh
#
# You cover: Logistic Regression, Ridge/Lasso/ElasticNet, LDA, QDA.
#
# Read for writing: Chang et al. (2024), Malakauskas and Lakstutiene (2021), Kpatcha (2025).
#
# Algorithm — Logistic Regression:
# Input: processed training data. Step 1: Fit linear model to estimate default probability. Step 2: Convert to probability using logistic function. Step 3: Predict on validation. Step 4: Choose cutoff using validation profit. Output: default probability, approval decision, evaluation metrics.
#
# You can tune: C, l1_ratio, reg_param, Optuna search space. Do not change: data loading, target, split, profit function, threshold tuning, final test.
#
# When done, send: model result CSV, threshold table CSV, hyperparameter table, best model summary, validation profit, AUC, Brier, approval rate, approved default rate, and what the result means for a lender.

#%% 19. Hai Anh models — Logistic / LDA / QDA

RUN_HAI_ANH_MODELS = False
RUN_HAI_ANH_OPTUNA = False
M_DIR = FINAL_OUT / "model_results"; M_DIR.mkdir(parents=True, exist_ok=True)
hah_result_path = M_DIR / f"hai_anh_results_{FEATURE_MODE}.csv"
hah_threshold_path = M_DIR / f"hai_anh_threshold_tables_{FEATURE_MODE}.csv"
ha_result_path = M_DIR / f"hai_an_results_{FEATURE_MODE}.csv"
ha_threshold_path = M_DIR / f"hai_an_threshold_tables_{FEATURE_MODE}.csv"
huy_result_path = M_DIR / f"huyen_anh_results_{FEATURE_MODE}.csv"
huy_threshold_path = M_DIR / f"huyen_anh_threshold_tables_{FEATURE_MODE}.csv"

hah_results = []
hah_tt = []
owner = "Hai Anh"

if RUN_HAI_ANH_MODELS:
    for C in [0.1, 1.0, 10.0]:
        model = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE)
        pipe, prob, theory, tuned, tt = fit_predict_evaluate(f"Ridge_C{C}", model, preprocess_scaled, owner)
        hah_results.append(tuned)
        hah_tt.append(tt.assign(model=f"Ridge_C{C}", owner=owner))
        print(f"Ridge C={C}: Profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}")

    for C in [0.1, 0.5]:
        model = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=2000, random_state=RANDOM_STATE)
        pipe, prob, theory, tuned, tt = fit_predict_evaluate(f"Lasso_C{C}", model, preprocess_scaled, owner)
        hah_results.append(tuned)
        hah_tt.append(tt.assign(model=f"Lasso_C{C}", owner=owner))

    model = LogisticRegression(penalty="elasticnet", C=1.0, l1_ratio=0.3, solver="saga", max_iter=2000, random_state=RANDOM_STATE)
    pipe, prob, theory, tuned, tt = fit_predict_evaluate("ElasticNet", model, preprocess_scaled, owner)
    hah_results.append(tuned)
    hah_tt.append(tt.assign(model="ElasticNet", owner=owner))

    Xtd = preprocess_scaled_dense.fit_transform(X_train, y_train)
    Xvd = preprocess_scaled_dense.transform(X_valid)
    if hasattr(Xtd,"toarray"): Xtd = Xtd.toarray(); Xvd = Xvd.toarray()

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(Xtd, y_train)
    lda_p = lda.predict_proba(Xvd)[:, 1]
    lda_tt = tune_threshold_by_profit("LDA", y_valid, lda_p, amount_valid)
    lda_t = lda_tt.iloc[0].to_dict()
    lda_t["threshold_type"] = "validation_profit_tuned"; lda_t["owner"] = owner; lda_t["feature_mode"] = FEATURE_MODE
    hah_results.append(lda_t)
    hah_tt.append(lda_tt.assign(model="LDA", owner=owner))

    qda = QuadraticDiscriminantAnalysis(reg_param=0.2)
    qda.fit(Xtd, y_train)
    qda_p = qda.predict_proba(Xvd)[:, 1]
    qda_tt = tune_threshold_by_profit("QDA", y_valid, qda_p, amount_valid)
    qda_t = qda_tt.iloc[0].to_dict()
    qda_t["threshold_type"] = "validation_profit_tuned"; qda_t["owner"] = owner; qda_t["feature_mode"] = FEATURE_MODE
    hah_results.append(qda_t)
    hah_tt.append(qda_tt.assign(model="QDA", owner=owner))

    append_and_save_results(hah_result_path, hah_threshold_path, hah_results, hah_tt)
    print("Hai Anh core models done.")

    # Your own experiment — add another model or wider tuning here.
    EXPERIMENT_LOG.append({"owner":"Hai Anh","experiment":"Logistic/LDA/QDA baseline","what_changed":"C values [0.1,1,10], Lasso C=[0.1,0.5], ElasticNet C=1 l1=0.3, LDA lsqr, QDA reg=0.2","why":"Baseline logistic family sweep","result_file":f"hai_anh_results_{FEATURE_MODE}.csv"})

# Hai Anh Optuna
if RUN_HAI_ANH_OPTUNA and _HAS_OPTUNA:
    print("Hai Anh Optuna — tuning Logistic C and penalty...")
    N_TRIALS = 20
    def haih_obj(trial):
        penalty = trial.suggest_categorical("penalty", ["l2","elasticnet"])
        C = trial.suggest_float("C", 0.01, 20.0, log=True)
        kwargs = {"penalty":penalty,"C":C,"solver":"saga","max_iter":2000,"random_state":RANDOM_STATE}
        if penalty=="elasticnet": kwargs["l1_ratio"]=trial.suggest_float("l1_ratio",0.1,0.9)
        pipe = Pipeline([("preprocess",preprocess_scaled),("model",LogisticRegression(**kwargs))])
        pipe.fit(X_train,y_train); prob=pipe.predict_proba(X_valid)[:,1]
        tt = tune_threshold_by_profit("optuna",y_valid,prob,amount_valid)
        return tt["net_profit"].iloc[0]
    study = optuna.create_study(direction="maximize")
    study.optimize(haih_obj, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params.copy()
    penalty = best.get("penalty", "l2")
    C = best["C"]
    kwargs_refit = {"penalty": penalty, "C": C, "solver": "saga", "max_iter": 2000, "random_state": RANDOM_STATE}
    if penalty == "elasticnet":
        kwargs_refit["l1_ratio"] = best.get("l1_ratio", 0.3)
    model = LogisticRegression(**kwargs_refit)
    pipe, prob, theory, tuned, tt = fit_predict_evaluate("Logistic_Optuna", model, preprocess_scaled, owner)
    tuned["params"] = str(best)
    hah_results.append(tuned)
    hah_tt.append(tt.assign(model="Logistic_Optuna", owner=owner))
    append_and_save_results(hah_result_path, hah_threshold_path, hah_results, hah_tt)
    print(f"  Best: {best}, profit=${study.best_value:,.0f}")
elif RUN_HAI_ANH_OPTUNA: print("Optuna not installed — run: !pip install optuna")

#%% [markdown]
# # Model Section — Hai An
#
# You cover: Random Forest, Bagging, AdaBoost, HistGradientBoosting, XGBoost, LightGBM, Optuna tuning.
#
# Read for writing: Aruleba and Sun (2024), Emmanuel et al. (2024), Chang et al. (2024).
#
# Algorithm — Random Forest:
# Input: processed training data. Step 1: Draw bootstrap samples. Step 2: Train one tree per sample. Step 3: Average predicted probabilities. Step 4: Predict on validation. Step 5: Choose cutoff using validation profit. Output: default probability, approval decision, evaluation metrics.
#
# Run core models first. Then run XGBoost, LightGBM, Optuna. Start with 20 trials, increase if runtime is ok.
#
# You can tune: depth, n_estimators, leaf size, learning rate, XGBoost/LGBM params, Optuna search space. Do not change: data loading, target, split, profit function, threshold tuning, final test.
#
# When done, send: model result CSV, threshold table CSV, feature importance, hyperparameter table, best model summary, validation profit, AUC, Brier, approval rate, approved default rate, and what the result means for a lender.

#%% 20. Hai An models — Tree-family / HGB

RUN_HAI_AN_MODELS = False
RUN_HAI_AN_OPTUNA = False; RUN_HAI_AN_XGBOOST = False; RUN_HAI_AN_LIGHTGBM = False

ha_results = []
ha_tt = []
owner = "Hai An"

if RUN_HAI_AN_MODELS:
    for n, d, l, label in [(100,16,25,"RF_n100_d16_l25"),(100,16,50,"RF_n100_d16_l50"),(200,16,50,"RF_n200_d16_l50")]:
        model = RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_leaf=l, max_features=None, n_jobs=-1, random_state=RANDOM_STATE)
        pipe, prob, theory, tuned, tt = fit_predict_evaluate(label, model, preprocess_tree, owner)
        ha_results.append(tuned)
        ha_tt.append(tt.assign(model=label, owner=owner))
        print(f"{label}: Profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}")

    base_b = DecisionTreeClassifier(max_depth=16, min_samples_leaf=50, random_state=RANDOM_STATE)
    try: bag = BaggingClassifier(estimator=base_b, n_estimators=100, max_samples=0.7, bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE)
    except TypeError: bag = BaggingClassifier(base_estimator=base_b, n_estimators=100, max_samples=0.7, bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE)
    pipe, prob, theory, tuned, tt = fit_predict_evaluate("Bagging_n100", bag, preprocess_tree, owner)
    ha_results.append(tuned); ha_tt.append(tt.assign(model="Bagging_n100", owner=owner))

    base_a = DecisionTreeClassifier(max_depth=2, min_samples_leaf=200, random_state=RANDOM_STATE)
    try: boost = AdaBoostClassifier(estimator=base_a, n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE)
    except TypeError: boost = AdaBoostClassifier(base_estimator=base_a, n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE)
    pipe, prob, theory, tuned, tt = fit_predict_evaluate("AdaBoost", boost, preprocess_tree, owner)
    ha_results.append(tuned); ha_tt.append(tt.assign(model="AdaBoost", owner=owner))

    if _HAS_HGB:
        for mi, lr, ln, msl, l2, label in [(200,0.05,31,50,0.1,"HGB_i200"),(250,0.08,63,50,0.1,"HGB_i250")]:
            model = HistGradientBoostingClassifier(max_iter=mi, learning_rate=lr, max_leaf_nodes=ln, min_samples_leaf=msl, l2_regularization=l2, random_state=RANDOM_STATE)
            pipe, prob, theory, tuned, tt = fit_predict_evaluate(label, model, preprocess_hgb, owner)
            ha_results.append(tuned); ha_tt.append(tt.assign(model=label, owner=owner))

    append_and_save_results(ha_result_path, ha_threshold_path, ha_results, ha_tt)
    print("Hai An core models done.")

    # Feature importance
    try:
        rf_pipe = Pipeline([("preprocess",preprocess_tree),("model",RandomForestClassifier(n_estimators=100,max_depth=16,min_samples_leaf=25,max_features=None,n_jobs=-1,random_state=RANDOM_STATE))])
        rf_pipe.fit(X_train, y_train)
        fn = rf_pipe.named_steps["preprocess"].get_feature_names_out()
        fi = pd.DataFrame({"feature":fn,"importance":rf_pipe.named_steps["model"].feature_importances_}).sort_values("importance",ascending=False)
        fi.to_csv(M_DIR/"hai_an_feature_importance.csv", index=False)
    except Exception as e: print("Feature importance skipped:", e)

    # Your own experiment — add another tree/boosting setup here.
    EXPERIMENT_LOG.append({"owner":"Hai An","experiment":"RF/Bagging/AdaBoost/HGB sweep","what_changed":"RF depths 16, leaf 25/50, HGB iter=200/250","why":"Baseline tree-family sweep","result_file":f"hai_an_results_{FEATURE_MODE}.csv"})

# XGBoost
if RUN_HAI_AN_XGBOOST and _HAS_XGB:
    print("XGBoost...")
    model = make_xgb_classifier(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    pipe, prob, theory, tuned, tt = fit_predict_evaluate("XGBoost_n200_d8", model, preprocess_tree, owner)
    ha_results.append(tuned); ha_tt.append(tt.assign(model="XGBoost_n200_d8", owner=owner))
    append_and_save_results(ha_result_path, ha_threshold_path, ha_results, ha_tt)
    print(f"  XGBoost: Profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}")
elif RUN_HAI_AN_XGBOOST: print("XGBoost not installed — run: !pip install xgboost")

# LightGBM
if RUN_HAI_AN_LIGHTGBM and _HAS_LGB:
    print("LightGBM...")
    model = lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1)
    pipe, prob, theory, tuned, tt = fit_predict_evaluate("LightGBM_n200_d8", model, preprocess_tree, owner)
    ha_results.append(tuned); ha_tt.append(tt.assign(model="LightGBM_n200_d8", owner=owner))
    append_and_save_results(ha_result_path, ha_threshold_path, ha_results, ha_tt)
    print(f"  LightGBM: Profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}")
elif RUN_HAI_AN_LIGHTGBM: print("LightGBM not installed — run: !pip install lightgbm")

# Optuna for HGB
if RUN_HAI_AN_OPTUNA and _HAS_OPTUNA and _HAS_HGB:
    print("Hai An Optuna — tuning HGB...")
    N_TRIALS = 20
    def hai_obj(trial):
        lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        mi = trial.suggest_int("max_iter", 100, 400)
        ln = trial.suggest_int("max_leaf_nodes", 15, 127)
        msl = trial.suggest_int("min_samples_leaf", 20, 300)
        l2 = trial.suggest_float("l2_regularization", 0.01, 10.0, log=True)
        hgb = HistGradientBoostingClassifier(max_iter=mi, learning_rate=lr, max_leaf_nodes=ln, min_samples_leaf=msl, l2_regularization=l2, random_state=RANDOM_STATE)
        pipe = Pipeline([("preprocess",preprocess_hgb),("model",hgb)])
        pipe.fit(X_train, y_train); prob = pipe.predict_proba(X_valid)[:, 1]
        tt = tune_threshold_by_profit("optuna_hgb", y_valid, prob, amount_valid)
        return tt["net_profit"].iloc[0]
    study = optuna.create_study(direction="maximize")
    study.optimize(hai_obj, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    hgb = HistGradientBoostingClassifier(**best, random_state=RANDOM_STATE)
    pipe, prob, theory, tuned, tt = fit_predict_evaluate("HGB_Optuna", hgb, preprocess_hgb, owner)
    tuned["params"] = str(best)
    ha_results.append(tuned); ha_tt.append(tt.assign(model="HGB_Optuna", owner=owner))
    append_and_save_results(ha_result_path, ha_threshold_path, ha_results, ha_tt)
    print(f"  Best HGB: {best}, profit=${study.best_value:,.0f}")
elif RUN_HAI_AN_OPTUNA: print("Optuna or HGB not installed")

#%% [markdown]
# # Model Section — Huyen Anh
#
# You cover: Neural Network / MLP.
#
# Read for writing: Chang et al. (2024), Emmanuel et al. (2024), Sun et al. (2025).
#
# Algorithm — Neural Network:
# Input: scaled numeric features and encoded categorical features. Step 1: Pass through hidden layers. Step 2: Apply activation functions. Step 3: Output layer estimates default probability. Step 4: Update weights by minimizing classification loss. Step 5: Choose cutoff using validation profit. Output: default probability, approval decision, evaluation metrics.
#
# Run MLP baseline first. Then run the tuning block. Start with 20 trials, increase if runtime allows.
#
# You can tune: hidden layers, activation, alpha, learning rate, early stopping, Optuna or manual grid. Do not change: data loading, target, split, profit function, threshold tuning, final test.
#
# When done, send: model result CSV, threshold table CSV, architecture table, training settings table, loss curve, best model summary, validation profit, AUC, Brier, approval rate, approved default rate, and what the result means for a lender.

#%% 21. Huyen Anh models — Neural Network

RUN_HUYEN_ANH_MODELS = False
RUN_HUYEN_ANH_OPTUNA = False

huy_results = []
huy_tt = []
owner = "Huyen Anh"

if RUN_HUYEN_ANH_MODELS:
    mlp = MLPClassifier(hidden_layer_sizes=(128,64,32), activation="relu", solver="adam", alpha=0.01,
                         early_stopping=True, validation_fraction=0.10, n_iter_no_change=10, max_iter=300, random_state=RANDOM_STATE)
    pipe, prob, theory, tuned, tt = fit_predict_evaluate("MLP_128x64x32", mlp, preprocess_scaled, owner)
    huy_results.append(tuned)
    huy_tt.append(tt.assign(model="MLP_128x64x32", owner=owner))
    print(f"NN: Profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}")

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(pipe.named_steps["model"].loss_curve_); ax.set_title("NN Training Loss")
    plt.tight_layout(); plt.savefig(M_DIR/"huyen_anh_loss_curve.png", dpi=120); plt.close()

    append_and_save_results(huy_result_path, huy_threshold_path, huy_results, huy_tt)
    print("Huyen Anh core model done.")

    # Your own experiment — add another architecture or tuning here.
    EXPERIMENT_LOG.append({"owner":"Huyen Anh","experiment":"MLP baseline (128,64,32)","what_changed":"hidden=(128,64,32), alpha=0.01, max_iter=300","why":"Baseline neural network","result_file":f"huyen_anh_results_{FEATURE_MODE}.csv"})

# Huyen Anh Optuna
if RUN_HUYEN_ANH_OPTUNA and _HAS_OPTUNA:
    print("Huyen Anh Optuna — tuning MLP...")
    N_TRIALS = 20
    def huy_obj(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden = tuple(trial.suggest_int(f"units_l{i}", 32, 256, step=32) for i in range(n_layers))
        alpha = trial.suggest_float("alpha", 1e-4, 0.1, log=True)
        lr = trial.suggest_float("learning_rate_init", 1e-4, 0.01, log=True)
        mlp = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                             alpha=alpha, learning_rate_init=lr, early_stopping=True,
                             validation_fraction=0.10, n_iter_no_change=10, max_iter=300, random_state=RANDOM_STATE)
        pipe = Pipeline([("preprocess",preprocess_scaled),("model",mlp)])
        pipe.fit(X_train, y_train); prob = pipe.predict_proba(X_valid)[:, 1]
        tt = tune_threshold_by_profit("optuna_mlp", y_valid, prob, amount_valid)
        return tt["net_profit"].iloc[0]
    study = optuna.create_study(direction="maximize")
    study.optimize(huy_obj, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    # Rebuild best model
    n_layers = best.get("n_layers", 2)
    hidden = tuple(best[f"units_l{i}"] for i in range(n_layers))
    alpha = best.get("alpha", 0.01)
    lr = best.get("learning_rate_init", 0.001)
    mlp_best = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                              alpha=alpha, learning_rate_init=lr, early_stopping=True,
                              validation_fraction=0.10, n_iter_no_change=10, max_iter=300, random_state=RANDOM_STATE)
    pipe, prob, theory, tuned, tt = fit_predict_evaluate("MLP_Optuna", mlp_best, preprocess_scaled, owner)
    tuned["params"] = str(best)
    huy_results.append(tuned); huy_tt.append(tt.assign(model="MLP_Optuna", owner=owner))
    append_and_save_results(huy_result_path, huy_threshold_path, huy_results, huy_tt)
    print(f"  Best MLP: {best}, profit=${tuned['net_profit']:,.0f}, AUC={tuned['auc']:.4f}")
elif RUN_HUYEN_ANH_OPTUNA: print("Optuna not installed — run: !pip install optuna")

# Save experiment log
EXPERIMENT_LOG = [e for e in EXPERIMENT_LOG if isinstance(e,dict)]
if EXPERIMENT_LOG:
    pd.DataFrame(EXPERIMENT_LOG).to_csv(FINAL_OUT/"experiment_log.csv", index=False)

#%% [markdown]
# # Validation Leaderboard
#
# Run this after model CSVs are saved. Combines all model results, ranks by validation net profit after threshold tuning.
# This is for model selection — it is not the final test result.

#%% 22. Combine model results

all_dfs = []
for fp in sorted(M_DIR.glob("*_results_*.csv")):
    mdf = pd.read_csv(fp)
    mode_from = "sba_plus_external" if "sba_plus_external" in fp.name else "sba_only" if "sba_only" in fp.name else "unknown"
    if "owner" not in mdf.columns:
        for o in ["Hai Anh","Hai An","Huyen Anh"]:
            if o.lower().replace(" ","_") in fp.name: mdf["owner"] = o
    if "feature_mode" not in mdf.columns: mdf["feature_mode"] = mode_from
    all_dfs.append(mdf)

if all_dfs:
    lb = pd.concat(all_dfs, ignore_index=True)
    lb = lb[lb["threshold_type"].str.contains("tuned", na=False)]
    lb = lb.sort_values("net_profit", ascending=False).reset_index(drop=True)
    lb.to_csv(M_DIR/f"full_model_leaderboard_{FEATURE_MODE}.csv", index=False)
    lb.to_csv(M_DIR/"full_model_leaderboard_all.csv", index=False)
    cols = ["model","owner","feature_mode","threshold_default","auc","brier","net_profit","approval_rate","approved_default_rate","runtime_seconds"]
    print("\nValidation leaderboard:")
    print(lb[[c for c in cols if c in lb.columns]].to_string(index=False))

#%% [markdown]
# # Final Test — Protected
#
# Do not run until: all EDA done, all model owners finish their runs, final model selected, final validation threshold frozen.
# I will run this once after we freeze everything.

#%% 23. Final test placeholders — DO NOT RUN

def add_engineered_features_for_model(data):
    """Recreate all built-in engineered features on any dataframe (dev or test)."""
    data = data.copy()
    if "NAICS_sector_str" not in data.columns:
        data["NAICS_sector_str"] = pd.to_numeric(data["NAICS_sector"],errors="coerce").astype("Int64").astype(str)

    tourism_codes = ["44","45","71","72"]
    coastal_list = ["AL","AK","CA","CT","DE","FL","GA","HI","LA","ME","MD","MA","MS","NH","NJ","NY","NC","OR","RI","SC","TX","VA","WA"]
    ds_codes = ["23","44","45","48","49","51","71","72"]
    supply_codes = ["23","31","32","33","42","48","49"]
    consumer_codes = ["44","45","71","72"]

    # Hai An — industry / housing
    if "state_hpi_negative_flag" in data.columns:
        is_re = data["NAICS_sector_str"] == "53"
        is_const = data["NAICS_sector_str"] == "23"
        hpi_down = data["state_hpi_negative_flag"] == 1
        data["real_estate_industry_hpi"] = (is_re & hpi_down).astype(int)
        data["construction_hpi"] = (is_const & hpi_down).astype(int)
    if "high_inflation_flag_lag1" in data.columns:
        data["supply_chain_inflation"] = (data["NAICS_sector_str"].isin(supply_codes) & (data["high_inflation_flag_lag1"]==1)).astype(int)

    # Hai Anh — local economy
    if "high_inflation_flag_lag1" in data.columns and "NoEmp" in data.columns:
        data["inflation_micro_business"] = ((data["high_inflation_flag_lag1"]==1) & (data["NoEmp"]<=5)).astype(int)
    if "oil_shock_flag_lag1" in data.columns:
        data["oil_shock_supply_chain"] = ((data["oil_shock_flag_lag1"]==1) & data["NAICS_sector_str"].isin(supply_codes)).astype(int)
    if "real_income_decline_flag_lag1" in data.columns:
        data["income_decline_consumer_sector"] = ((data["real_income_decline_flag_lag1"]==1) & data["NAICS_sector_str"].isin(consumer_codes)).astype(int)
    if "state_bank_failures_12m" in data.columns:
        data["recent_bank_failure_flag"] = (data["state_bank_failures_12m"].fillna(0)>0).astype(int)

    # Huyen Anh — disaster / geography
    data["tourism_industry_flag"] = data["NAICS_sector_str"].isin(tourism_codes).astype(int)
    data["coastal_state_flag"] = data["State"].isin(coastal_list).astype(int)
    data["disaster_sensitive_sector_flag"] = data["NAICS_sector_str"].isin(ds_codes).astype(int)
    if "fema_hurricane_12m" in data.columns:
        data["tourism_hurricane"] = data["tourism_industry_flag"] * data["fema_hurricane_12m"]
        data["coastal_hurricane"] = data["coastal_state_flag"] * data["fema_hurricane_12m"]
        data["disaster_sensitive_hurricane"] = data["disaster_sensitive_sector_flag"] * data["fema_hurricane_12m"]
    return data

RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    print("FINAL TEST: Only I should run this after final model and threshold are frozen.")

#%% 24. Profit curve helper

def plot_profit_curve(model_name, prob_default, y_true, amount_series):
    df_p = pd.DataFrame({"pd":prob_default,"y":np.asarray(y_true),"amt":np.asarray(amount_series)})
    df_p = df_p.sort_values("pd").reset_index(drop=True)
    df_p["cum_profit"] = loan_profit_vector(df_p["y"],np.ones(len(df_p),dtype=bool),df_p["amt"]).cumsum()
    df_p["approval_depth"] = np.arange(1,len(df_p)+1)/len(df_p)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(df_p["approval_depth"],df_p["cum_profit"])
    ax.axvline(df_p["cum_profit"].idxmax()/len(df_p),ls="--",color="red")
    ax.set_title(f"Profit Curve — {model_name}"); ax.set_xlabel("Approval Depth"); ax.set_ylabel("Cumulative Profit")
    plt.tight_layout(); plt.savefig(M_DIR/"profit_curve.png",dpi=120); plt.close()
    return df_p

print("Profit curve helper ready.")

#%% 25. End checklist

print(f"Dataset loaded: {FILE_TYPE}")
print(f"Dev/test split: dev={len(df_dev):,}, test={len(df_final_test):,}")
print(f"Modeling data: full dev set, n={len(df_model):,}")
print(f"Baseline EDA: done ({OUT_DIR})")
print(f"Hai An external EDA: {'done' if (OUT_HA/'construction_hpi.png').exists() else 'not run'}")
print(f"Hai Anh external EDA: {'done' if (OUT_HAH/'stress_score.png').exists() else 'not run'}")
print(f"Huyen Anh external EDA: {'done' if (OUT_HUY/'fema_disaster_type_default_rates.csv').exists() else 'not run'}")
print(f"Feature notes: {'saved' if (FINAL_OUT/'feature_notes.csv').exists() else 'not saved'}")
print(f"Experiment log: {'saved' if (FINAL_OUT/'experiment_log.csv').exists() else 'not saved'}")
print(f"Hai Anh models: {'run' if RUN_HAI_ANH_MODELS else 'not run'}")
print(f"Hai An models: {'run' if RUN_HAI_AN_MODELS else 'not run'}")
print(f"Huyen Anh models: {'run' if RUN_HUYEN_ANH_MODELS else 'not run'}")
print(f"Leaderboard: {'saved' if (M_DIR/'full_model_leaderboard_all.csv').exists() else 'not yet'}")
print(f"Final test: PROTECTED")
print("Done.")
