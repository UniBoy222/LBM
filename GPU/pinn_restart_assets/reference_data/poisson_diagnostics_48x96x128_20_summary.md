# Poisson Residual Diagnostics Summary

## Aggregate

| metric | value |
| --- | --- |
| steps | 20 |
| tolerance | 0.001 |
| max_iterations | 1000 |
| converged_steps | 2 |
| maxed_out_steps | 18 |
| slow_or_stalled_tail_steps | 18 |
| mean_final_iteration | 960 |
| mean_final_relative_error | 0.0184 |
| median_late_ratio | 0.9524 |
| median_log10_drop | 0.5416 |

## Interpretation

- Many steps hit the 1000-iteration cap; reducing launch overhead alone cannot solve this.
- Late residual decay is slow, so the next candidate should target fixed-point acceleration or a coarse-grid correction.

## Per-Step Summary

| step | final iter | final rel error | late ratio | class |
| --- | ---: | ---: | ---: | --- |
| 1 | 1000 | 0.0723 | 0.8708 | stalled_tail |
| 2 | 1000 | 0.02681 | 0.8434 | slow_tail |
| 3 | 1000 | 0.06989 | 0.971 | stalled_tail |
| 4 | 1000 | 0.08057 | 0.9075 | stalled_tail |
| 5 | 1000 | 0.03368 | 0.9148 | stalled_tail |
| 6 | 1000 | 0.01245 | 0.966 | stalled_tail |
| 7 | 1000 | 0.01305 | 0.9288 | stalled_tail |
| 8 | 1000 | 0.01244 | 0.9325 | stalled_tail |
| 9 | 1000 | 0.008211 | 0.9529 | stalled_tail |
| 10 | 1000 | 0.006962 | 0.9784 | stalled_tail |
| 11 | 1000 | 0.008047 | 0.9679 | stalled_tail |
| 12 | 1000 | 0.007604 | 0.9703 | stalled_tail |
| 13 | 1000 | 0.004976 | 0.9753 | stalled_tail |
| 14 | 1000 | 0.002543 | 0.9772 | stalled_tail |
| 15 | 1000 | 0.001604 | 0.9337 | stalled_tail |
| 16 | 1000 | 0.001805 | 0.9518 | stalled_tail |
| 17 | 1000 | 0.001918 | 0.9644 | stalled_tail |
| 18 | 1000 | 0.001368 | 0.9755 | stalled_tail |
| 19 | 800 | 9.6164e-04 | 0.9465 | converged |
| 20 | 400 | 7.6451e-04 | 0.7593 | converged |
