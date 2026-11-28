### `sft_fulltrain_dev` vs `arm_counterfactual`

| 格 | A | B | Δ(A−B) | n |
|---|---:|---:|---:|---:|
| easy_consistent | 0.965 | 0.923 | +0.042 | 312 |
| easy_conflict | 0.580 | 0.596 | -0.016 | 312 |
| hard_consistent | 0.939 | 0.913 | +0.026 | 312 |
| hard_conflict | 0.558 | 0.577 | -0.019 | 312 |

- McNemar：n=1248，仅 A 对 63，仅 B 对 53，delta=+0.008，p=0.4035
