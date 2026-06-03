# Complete Model Ranking — SBA Loan Default Project

All results on shared feature set v1 (24 numeric + 7 categorical), 50,000-row sample, random_state=1, 60/20/20 split.
**Updated 2026-06-03 with Hai An Shared V1 Rerun results.**

Approve-all baseline: **$30,457,862**
Deny-all baseline: **$0**

---

## Full Ranking (tuned thresholds, validation set only)

| # | Model | AUC | Net Profit | Threshold | Approval | Default Rate | Runtime | Source |
|---|-------|-----|-----------|-----------|----------|-------------|---------|--------|
| 1 | **HGB** 250 iter, lr=0.08, leaf=63 | 0.9712 | **$71,253,735** | 0.074 | 72.9% | 1.24% | 10s | Hai An V1 |
| 2 | HGB 200 iter, lr=0.05, leaf=31 | 0.9705 | $70,369,860 | 0.124 | 74.4% | 1.53% | 7s | Hai An V1 |
| 3 | HGB 200 iter, lr=0.05, leaf=31 | 0.9705 | $70,022,621 | 0.110 | 73.5% | 1.47% | 32s | Benchmark |
| 4 | HGB 200 iter, lr=0.03, leaf=31 | 0.9694 | $69,722,523 | 0.154 | 75.7% | 1.82% | 7s | Hai An V1 |
| 5 | HGB 100 iter, lr=0.05, leaf=31 | 0.9683 | $69,493,734 | 0.164 | 75.9% | 1.98% | 4s | Hai An V1 |
| 6 | HGB 200 iter, lr=0.05, leaf=15 | 0.9682 | $68,979,210 | 0.184 | 76.7% | 2.10% | 4s | Hai An V1 |
| 7 | **RF** n=100, depth=16, leaf=25 | 0.9657 | $68,860,349 | 0.203 | 76.1% | 2.16% | 4s | Hai An V1 |
| 8 | RF n=100, depth=16, leaf=25, feat=None | 0.9649 | $68,901,499 | 0.151 | 73.1% | 1.57% | 12s | Benchmark |
| 9 | RF n=100, depth=12, leaf=25, feat=None | 0.9642 | $67,980,630 | 0.138 | 72.4% | 1.64% | 11s | Benchmark |
| 10 | RF n=100, depth=16, leaf=50, feat=None | 0.9617 | $67,738,552 | 0.201 | 74.9% | 2.20% | 10s | Benchmark |
| 11 | RF n=100, depth=16, leaf=50 | 0.9626 | $67,706,029 | 0.198 | 74.7% | 2.10% | 3s | Hai An V1 |
| 12 | RF n=200, depth=16, leaf=50, feat=None | 0.9619 | $67,558,180 | 0.216 | 75.7% | 2.36% | 20s | Benchmark |
| 13 | RF n=200, depth=16, leaf=50 | 0.9627 | $67,534,661 | 0.213 | 75.5% | 2.28% | 6s | Hai An V1 |
| 14 | RF n=200, depth=None, leaf=50, feat=None | 0.9619 | $67,529,167 | 0.217 | 75.7% | 2.38% | 18s | Benchmark |
| 15 | **Bagging** n=100, d=16, leaf=50 | 0.9608 | $67,366,143 | 0.184 | 73.9% | 2.06% | 5s | Hai An V1 |
| 16 | Bagging n=100, d=16, leaf=50, samp=0.7 | 0.9599 | $67,186,796 | 0.184 | 73.4% | 2.06% | 9s | Benchmark |
| 17 | Decision Tree depth=16, leaf=50 | 0.9516 | $66,913,274 | 0.110 | 70.6% | 1.67% | 1s | Benchmark |
| 18 | Bagging n=100, d=12, leaf=50, samp=0.7 | 0.9594 | $66,716,732 | 0.193 | 74.1% | 2.21% | 8s | Benchmark |
| 19 | RF n=100, d=16, leaf=50, feat=0.5 | 0.9593 | $66,639,715 | 0.184 | 73.5% | 2.22% | 5s | Benchmark |
| 20 | Bagging n=50, d=12, leaf=50, samp=0.7 | 0.9589 | $66,555,414 | 0.192 | 74.1% | 2.24% | 5s | Benchmark |
| 21 | Decision Tree depth=12, leaf=50 | 0.9529 | $65,798,479 | 0.110 | 71.2% | 1.91% | 1s | Benchmark |
| 22 | DT depth=16, leaf=100 | 0.9477 | $64,954,960 | 0.209 | 75.9% | 3.21% | 1s | Benchmark |
| 23 | DT depth=12, leaf=100 | 0.9470 | $64,954,960 | 0.209 | 75.9% | 3.21% | 1s | Benchmark |
| 24 | Bagging n=100, d=16, leaf=100, samp=0.8 | 0.9509 | $64,379,160 | 0.143 | 68.8% | 1.86% | 7s | Benchmark |
| 25 | Bagging n=50, d=12, leaf=100, samp=0.7 | 0.9486 | $64,142,995 | 0.118 | 67.1% | 1.64% | 4s | Benchmark |
| 26 | DT depth=8, leaf=50 | 0.9452 | $63,519,458 | 0.168 | 74.7% | 3.13% | 1s | Benchmark |
| 27 | AdaBoost d=2, leaf=200 | 0.9054 | $55,092,816 | 0.164 | 64.4% | 2.95% | 5s | Hai An V1 |
| 28 | AdaBoost d=2, leaf=200, n=100 | 0.9050 | $55,005,762 | 0.279 | 76.0% | 4.55% | 6s | Benchmark |
| 29 | **KNN** k=51, distance | 0.8601 | $53,832,218 | 0.184 | 66.5% | 5.54% | 8s | Hai An V1 |
| 30 | **NN** (128,64,32), α=0.01 | 0.9190 | $60,893,607 | 0.076 | 71.6% | 3.84% | 3s | Benchmark |
| 31 | NN (128,64,32), α=0.05 | 0.9205 | $60,071,849 | 0.193 | 77.6% | 4.74% | 3s | Benchmark |
| 32 | NN (128,64,32), α=0.001 | 0.9136 | $59,629,680 | 0.052 | 70.1% | 3.75% | 3s | Benchmark |
| 33 | NN (128,64), α=0.01 | 0.9133 | $58,884,897 | 0.080 | 70.8% | 3.72% | 3s | Benchmark |
| 34 | NN (64,32), α=0.01 | 0.9103 | $58,807,686 | 0.135 | 73.3% | 4.38% | 3s | Benchmark |
| 35 | RF n=100, d=16, leaf=50, feat=sqrt | 0.9110 | $58,092,592 | 0.251 | 76.0% | 4.52% | 4s | Benchmark |
| 36 | **Lasso C=0.1** | 0.8380 | $50,721,996 | 0.263 | 75.6% | 6.85% | 27s | Hai Anh Cleaned |
| 37 | **Logistic Ridge C=10.0** | 0.8390 | $50,404,027 | 0.247 | 74.1% | 6.76% | 21s | Benchmark |
| 38 | Logistic ElasticNet C=1.0, l1=0.3 | 0.8389 | $50,382,675 | 0.247 | 74.1% | 6.75% | 73s | Benchmark |
| 39 | Logistic Lasso C=0.5 | 0.8386 | $50,211,085 | 0.226 | 71.9% | 6.51% | 56s | Benchmark |
| 40 | **LDA** lsqr, shrinkage=auto | 0.8185 | $49,674,878 | 0.164 | 62.0% | 5.88% | 0.3s | Hai Anh Cleaned |
| 41 | **QDA** reg_param=0.2 | 0.8016 | $44,227,842 | 0.135 | 61.3% | 7.37% | 0.3s | Benchmark |
| — | Approve All | — | $30,457,862 | — | 100% | 17.56% | — | — |
| — | Deny All | — | $0 | — | 0% | — | — | — |

---

## Best Model per Family

| Family | Best Model | Profit | AUC | Owner |
|--------|-----------|--------|-----|-------|
| **HGB** | 250 iter, lr=0.08, leaf=63 | **$71,253,735** | 0.9712 | Hai An |
| Random Forest | n=100, depth=16, leaf=25 | $68,860,349 | 0.9657 | Hai An |
| Decision Tree | depth=16, leaf=50 | $66,913,274 | 0.9516 | Hai An |
| Neural Network | (128,64,32), α=0.01 | $60,893,607 | 0.9190 | Huyen Anh |
| AdaBoost | depth=2, leaf=200 | $55,092,816 | 0.9054 | Hai An |
| KNN | k=51, distance | $53,832,218 | 0.8601 | Hai An |
| Logistic | Lasso C=0.1 | $50,721,996 | 0.8380 | Hai Anh |
| LDA | lsqr, shrinkage=auto | $49,674,878 | 0.8185 | Hai Anh |
| QDA | reg=0.2 | $44,227,842 | 0.8016 | Hai Anh |

---

## Key Observations (updated)

1. **HGB at $71.3M** — with full v1 features, Hai An's HGB (250 iter, lr=0.08, leaf=63) becomes the new champion, +$1.3M above the official benchmark best.
2. **v1 features add ~$3M to HGB** — going from the old feature set (17 numeric) to full v1 (24 numeric) lifted HGB from $68.3M → $71.3M.
3. **Tree-family occupies top 7 spots** — nonlinear patterns dominate.
4. **NN is steady at $60.9M** — strong AUC but conservative threshold limits profit.
5. **Lasso C=0.1 edges Ridge** on the cleaned run — $50.7M vs $50.4M. Minor difference.
6. **`max_features=None` always wins** — every feature contributes to tree models.

---

## Source Files for Teammates

| File | For | Description |
|------|-----|-------------|
| `Hai An Shared V1 Rerun.py` | Hai An | His KNN + tree-family code upgraded with full v1 features |
| `Hai Anh Cleaned Preview.py` | Hai Anh | Her logistic/LDA/QDA code, cleaned (no Block 21, no hardcoded path) |
| `Huyen Anh Cleaned Main Workflow.py` | Huyen Anh | Her NN code from branch, RUN flags set to False |

---

## Reproducibility Check (all 3 teammate files)

| Check | Hai An | Hai Anh | Huyen Anh |
|---|---|---|---|
| `DATA_PATH = "SBAnational.csv"` | ✅ | ✅ | ✅ |
| `RANDOM_STATE = 1` | ✅ | ✅ | ✅ |
| `USE_WORKING_SAMPLE = True` | ✅ | ✅ | ✅ |
| `WORKING_SAMPLE_N = 50000` | ✅ | ✅ | ✅ |
| Train/valid/test = 60/20/20 | ✅ | ✅ | ✅ |
| `stratify=y` on both splits | ✅ | ✅ | ✅ |
| 24 numeric features | ✅ | ✅ | ✅ |
| 7 categorical features | ✅ | ✅ | ✅ |
| `DisbursementGross` guarded | ✅ | ✅ | ✅ |
| `log_DisbursementGross` in v1 | ✅ | ✅ | ✅ |
| Same profit function (5%/-25%) | ✅ | ✅ | ✅ |
| Same threshold tuning | ✅ | ✅ | ✅ |
| Test set untouched | ✅ | ✅ | ✅ |
| `RUN_FINAL_TEST = False` | ✅ | ✅ | ✅ |
| No hardcoded paths | ✅ | ✅ | ✅ |

**Verdict: All three files are cross-reproducible.** Running any file with the same random_state produces identical train/valid/test splits because they share the same feature set, sample logic, and stratified splitting.
