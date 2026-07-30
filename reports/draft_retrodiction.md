# Draft-board retrodiction (2023-2025)

Preseason boards from season N-1 data graded against realized season-N
PPR totals. `board_no_model_LOWER_BOUND` is the honest, leakage-free
grade; the leaky variant exists only to bound what model blending adds.

| variant | mean Spearman | mean top-24 hit rate |
|---|---|---|
| board_no_model_LOWER_BOUND | 0.7375 | 0.417 |
| board_leaky_model_REFERENCE_ONLY | 0.7334 | 0.431 |
| naive_lastyear_total | 0.7652 | 0.389 |
| naive_pergame_min6 | 0.7631 | 0.389 |

## 2023
- board_no_model_LOWER_BOUND: n=700, spearman=0.7402, top24 10/24
- board_leaky_model_REFERENCE_ONLY: n=700, spearman=0.7315, top24 10/24
- naive_lastyear_total: n=721, spearman=0.755, top24 9/24
- naive_pergame_min6: n=694, spearman=0.7515, top24 9/24

## 2024
- board_no_model_LOWER_BOUND: n=735, spearman=0.7399, top24 9/24
- board_leaky_model_REFERENCE_ONLY: n=735, spearman=0.7315, top24 9/24
- naive_lastyear_total: n=755, spearman=0.7671, top24 8/24
- naive_pergame_min6: n=722, spearman=0.7667, top24 8/24

## 2025
- board_no_model_LOWER_BOUND: n=714, spearman=0.7324, top24 11/24
- board_leaky_model_REFERENCE_ONLY: n=714, spearman=0.7372, top24 12/24
- naive_lastyear_total: n=727, spearman=0.7736, top24 11/24
- naive_pergame_min6: n=701, spearman=0.771, top24 11/24
