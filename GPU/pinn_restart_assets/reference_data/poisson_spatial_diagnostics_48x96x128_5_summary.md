# Poisson Residual Diagnostics Summary

## Aggregate

| metric | value |
| --- | --- |
| steps | 5 |
| tolerance | 0.001 |
| max_iterations | 1000 |
| converged_steps | 0 |
| maxed_out_steps | 5 |
| slow_or_stalled_tail_steps | 5 |
| mean_final_iteration | 1000 |
| mean_final_relative_error | 0.05665 |
| median_late_ratio | 0.9075 |
| median_log10_drop | 0.5698 |
| median_final_block_low_frequency_fraction | 0.9936 |
| median_mean_block_low_frequency_fraction | 0.9779 |

## Interpretation

- Many steps hit the 1000-iteration cap; reducing launch overhead alone cannot solve this.
- Late residual decay is slow, so the next candidate should target fixed-point acceleration or a coarse-grid correction.
- Block residual cancellation is weak, suggesting coarse-grid correction is plausible.

## Per-Step Summary

| step | final iter | final rel error | late ratio | block low freq | class |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 1000 | 0.0723 | 0.8708 | 1 | stalled_tail |
| 2 | 1000 | 0.02681 | 0.8434 | 0.9867 | slow_tail |
| 3 | 1000 | 0.06989 | 0.971 | 0.9936 | stalled_tail |
| 4 | 1000 | 0.08057 | 0.9075 | 0.9967 | stalled_tail |
| 5 | 1000 | 0.03368 | 0.9148 | 0.9927 | stalled_tail |
