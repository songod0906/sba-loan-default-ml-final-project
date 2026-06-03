# Complete Model Ranking — SBA Loan Default Project

All results on shared feature set v1 (23 numeric + 7 categorical).
50,000-row sample, `random_state=1`, 60/20/20 split.
**Updated 2026-06-03: log_DisbursementGross removed (DisbursementGross is post-approval leakage).**

Approve-all baseline: **$30,457,862**
Deny-all baseline: **$0**

---

## Top 10 (tuned thresholds, validation set only)

| # | Model | AUC | Net Profit | Threshold | Approval | Default Rate | Runtime | Owner |
|---|-------|-----|-----------|-----------|----------|-------------|---------|-------|
| 1 | **HGB** 250 iter, lr=0.08, leaf=63 | 0.9704 | **$71,535,187** | 0.104 | 74.9% | 1.48% | 11s | Hai An |
| 2 | HGB 200 iter, lr=0.05, leaf=31 | 0.9705 | $70,860,641 | 0.124 | 74.3% | 1.53% | 8s | Hai An |
| 3 | HGB 200 iter, lr=0.03, leaf=31 | 0.9693 | $69,525,823 | 0.188 | 77.2% | 2.14% | 8s | Hai An |
| 4 | HGB 200 iter, lr=0.05, leaf=15 | 0.9684 | $69,521,320 | 0.094 | 70.2% | 1.32% | 5s | Hai An |
| 5 | HGB 100 iter, lr=0.05, leaf=31 | 0.9679 | $69,195,918 | 0.169 | 76.1% | 2.06% | 4s | Hai An |
| 6 | **RF** n=100, depth=16, leaf=25 | 0.9659 | $68,839,653 | 0.159 | 73.7% | 1.70% | 4s | Hai An |
| 7 | RF n=100, depth=16, leaf=50 | 0.9628 | $67,825,944 | 0.198 | 74.8% | 2.09% | 3s | Hai An |
| 8 | RF n=200, depth=16, leaf=50 | 0.9628 | $67,562,577 | 0.213 | 75.5% | 2.28% | 6s | Hai An |
| 9 | Bagging n=100, d=16, leaf=50 | 0.9608 | $67,256,513 | 0.184 | 73.8% | 2.06% | 5s | Hai An |
| 10 | **NN** (128,64,32), α=0.01 | 0.9177 | $60,297,144 | 0.069 | 67.5% | 3.19% | 3s | Huyen Anh |

## Best per Family

| Family | Model | Profit | AUC | Owner |
|--------|-------|--------|-----|-------|
| **HGB** | 250 iter, lr=0.08, leaf=63 | **$71,535,187** | 0.9704 | Hai An |
| Random Forest | n=100, depth=16, leaf=25 | $68,839,653 | 0.9659 | Hai An |
| Neural Network | (128,64,32), α=0.01 | $60,297,144 | 0.9177 | Huyen Anh |
| AdaBoost | depth=2, leaf=200 | $55,092,816 | 0.9054 | Hai An |
| KNN | k=51, distance | $54,329,264 | 0.8620 | Hai An |
| Logistic Ridge | C=1.0 | $51,436,742 | 0.8402 | Hai Anh |
| LDA | lsqr, shrinkage=auto | $50,135,737 | 0.8169 | Hai Anh |
| QDA | reg=0.2 | $44,469,480 | 0.8027 | Hai Anh |

## Full Logistic/LDA/QDA Ranking

| Model | Profit | AUC | Threshold | Approval |
|-------|--------|-----|-----------|----------|
| Ridge C=1.0 | $51,436,742 | 0.8402 | 0.223 | 71.6% |
| ElasticNet | $51,369,208 | 0.8400 | 0.218 | 70.8% |
| Ridge C=10.0 | $51,131,084 | 0.8400 | 0.223 | 71.5% |
| Lasso C=0.5 | $51,125,943 | 0.8398 | 0.213 | 70.3% |
| Lasso C=0.1 | $51,096,975 | 0.8376 | 0.263 | 75.5% |
| Ridge C=0.1 | $51,037,368 | 0.8397 | 0.193 | 67.3% |
| LDA | $50,135,737 | 0.8169 | 0.169 | 62.6% |
| QDA reg=0.2 | $44,469,480 | 0.8027 | 0.139 | 61.3% |

## Key Observations

1. **HGB dominates** — all top 5 spots. Removing `log_DisbursementGross` had minimal impact (actually improved $71.3M → $71.5M).
2. **RF is strong** — depth=16/leaf=25 is the sweet spot.
3. **NN slightly drops** — from $60.9M to $60.3M without log_DisbursementGross.
4. **Logistic improves** — Ridge from $50.4M to $51.4M without the post-approval feature.
5. **DisbursementGross is correctly excluded** — it is the amount disbursed after approval/funding, and using it (even log-transformed) would be leakage.
6. **Feature count: 23 numeric + 7 categorical** — down from 24 numeric.
7. **All teammate files are cross-reproducible** — identical splits, features, scoring.
