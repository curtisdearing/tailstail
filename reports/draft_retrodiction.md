# Draft-board retrodiction (2023-2025)

Preseason boards from season N-1 data graded against realized season-N
PPR totals. `board_no_model_LOWER_BOUND` is the honest, leakage-free
grade; the leaky variant exists only to bound what model blending adds.

| variant | mean Spearman | mean top-24 hit rate |
|---|---|---|
| board_no_model_LOWER_BOUND | 0.6815 | 0.43 |
| board_leaky_model_REFERENCE_ONLY | 0.6731 | 0.403 |
| naive_lastyear_total | 0.7345 | 0.389 |
| naive_pergame_min6 | 0.6846 | 0.472 |

## 2023
- board_no_model_LOWER_BOUND: n=451, spearman=0.6952, top24 11/24
- board_leaky_model_REFERENCE_ONLY: n=451, spearman=0.6846, top24 10/24
- naive_lastyear_total: n=530, spearman=0.734, top24 9/24
- naive_pergame_min6: n=365, spearman=0.7079, top24 11/24

## 2024
- board_no_model_LOWER_BOUND: n=435, spearman=0.6689, top24 9/24
- board_leaky_model_REFERENCE_ONLY: n=435, spearman=0.6609, top24 8/24
- naive_lastyear_total: n=514, spearman=0.7225, top24 8/24
- naive_pergame_min6: n=383, spearman=0.6707, top24 11/24

## 2025
- board_no_model_LOWER_BOUND: n=437, spearman=0.6805, top24 11/24
- board_leaky_model_REFERENCE_ONLY: n=437, spearman=0.6737, top24 11/24
- naive_lastyear_total: n=520, spearman=0.7469, top24 11/24
- naive_pergame_min6: n=361, spearman=0.6752, top24 12/24
