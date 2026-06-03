# SBA Loan Default Modeling

## Dataset

`SBAnational.csv`

## Files

- `Main Workflow.py` = uniform team workflow (shared v1 features, all model blocks runnable)
- `Final Project Teaching.py` = learning/demo file
- `Teaching slide.pdf` = team teaching deck
- `COMPLETE_MODEL_RANKING.md` = full model ranking with detailed notes

## How to Run

1. Clone repo, place `SBAnational.csv` in the project folder.
2. `Main Workflow.py` uses Spyder-style `#%%` cells. Run Blocks 1–13 first for setup.
3. Set your `RUN_*` flag to `True` in your assigned model block (14A–14D).
4. Run your model block to see results on the shared validation set.
5. Run your CV/tuning block (15A–15D) for hyperparameter search.
6. Leaderboard builds automatically in Block 16.
7. **Do NOT run Block 19 (final test) until the team freezes features/models.**

## Team Workflow

| Owner | Model Family | Block | RUN flag |
|-------|-------------|-------|----------|
| Hai An | KNN | 14A | `RUN_KNN` |
| Hai An | Tree-family (DT/Bagging/RF/AdaBoost/HGB) | 14B | `RUN_TREE_MODELS` |
| Hai Anh | Logistic/LDA/QDA | 14C | `RUN_LOGIT_DA` |
| Huyen Anh | Neural Network | 14D | `RUN_NEURAL_NET` |

All teammates use the same 24 numeric + 7 categorical features (shared v1 set).
Results are comparable because everyone shares identical train/valid/test splits.

---

# Complete Model Ranking

All results on shared feature set v1, 50,000-row sample, `random_state=1`, 60/20/20 split.

Approve-all baseline: **$30,457,862**
Deny-all baseline: **$0**

## Top 10 (tuned thresholds, validation set only)

| # | Model | AUC | Net Profit | Threshold | Approval | Default Rate | Runtime | Owner |
|---|-------|-----|-----------|-----------|----------|-------------|---------|-------|
| 1 | **HGB** 250 iter, lr=0.08, leaf=63 | 0.9712 | **$71,253,735** | 0.074 | 72.9% | 1.24% | 10s | Hai An |
| 2 | HGB 200 iter, lr=0.05, leaf=31 | 0.9705 | $70,369,860 | 0.124 | 74.4% | 1.53% | 7s | Hai An |
| 3 | HGB 200 iter, lr=0.05, leaf=31 | 0.9705 | $70,022,621 | 0.110 | 73.5% | 1.47% | 32s | Benchmark |
| 4 | HGB 200 iter, lr=0.03, leaf=31 | 0.9694 | $69,722,523 | 0.154 | 75.7% | 1.82% | 7s | Hai An |
| 5 | HGB 100 iter, lr=0.05, leaf=31 | 0.9683 | $69,493,734 | 0.164 | 75.9% | 1.98% | 4s | Hai An |
| 6 | HGB 200 iter, lr=0.05, leaf=15 | 0.9682 | $68,979,210 | 0.184 | 76.7% | 2.10% | 4s | Hai An |
| 7 | **RF** n=100, depth=16, leaf=25 | 0.9657 | $68,860,349 | 0.203 | 76.1% | 2.16% | 4s | Hai An |
| 8 | RF n=100, depth=16, leaf=25 | 0.9649 | $68,901,499 | 0.151 | 73.1% | 1.57% | 12s | Benchmark |
| 9 | RF n=100, depth=12, leaf=25 | 0.9642 | $67,980,630 | 0.138 | 72.4% | 1.64% | 11s | Benchmark |
| 10 | RF n=100, depth=16, leaf=50 | 0.9617 | $67,738,552 | 0.201 | 74.9% | 2.20% | 10s | Benchmark |

## Best per Family

| Family | Model | Profit | AUC | Owner |
|--------|-------|--------|-----|-------|
| **HGB** | 250 iter, lr=0.08, leaf=63 | **$71,253,735** | 0.9712 | Hai An |
| Random Forest | n=100, depth=16, leaf=25 | $68,860,349 | 0.9657 | Hai An |
| Decision Tree | depth=16, leaf=50 | $66,913,274 | 0.9516 | Hai An |
| Neural Network | (128,64,32), α=0.01 | $60,893,607 | 0.9190 | Huyen Anh |
| AdaBoost | depth=2, leaf=200 | $55,092,816 | 0.9054 | Hai An |
| KNN | k=51, distance | $53,832,218 | 0.8601 | Hai An |
| Logistic | Lasso C=0.1 | $50,721,996 | 0.8380 | Hai Anh |
| LDA | lsqr, shrinkage=auto | $49,674,878 | 0.8185 | Hai Anh |
| QDA | reg=0.2 | $44,227,842 | 0.8016 | Hai Anh |

## Key Observations

1. **HGB dominates** — all top 6 spots. Best threshold around 0.07–0.16.
2. **RF is strong** — min_samples_leaf=25 beats leaf=50 by ~$1.2M.
3. **NN is competitive** — AUC 0.919 but conservative threshold limits profit.
4. **Logistic is solid** — Ridge C=10.0 at $50.4M is predictable and explainable.
5. **`max_features=None` always wins** — every feature contributes to tree models.
6. **v1 features matter** — upgrading from old 17-feature set to full 24-feature v1 added ~$3M to HGB.
7. **All 3 teammate files are cross-reproducible** — identical splits, features, scoring.
