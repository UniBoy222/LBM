# Strict unbounded Poisson baseline contract

All production pressure-solver entry points in this tree use the contract below:

- split Poisson and split pressure boundary;
- `while (true)` iteration until convergence;
- residual check interval: `100`;
- convergence tolerance: `1e-3` with strict `residual < tolerance` acceptance;
- `poisson_iteration_limit=0`, meaning unbounded;
- immediate failure on non-finite source, pressure, gauge, or residual values;
- explicit overflow checks for the per-step Poisson iteration counter; CUDA also
  checks graph-segment multiplication, accumulated iterations, and its time-step
  counter.

This applies to `GPU/lbm_gpu --mode gpu`, `GPU/lbm_gpu --mode cpu`, and the
standalone CPU solver.  The legacy CPU NN pressure initializer is disabled in
the strict baseline.  A future learned warm-start must use a separate explicit
and auditable entry point.

`max_time_steps` controls the number of outer physical LBM steps and is not a
Poisson iteration cap.  The checked-in examples use 20 outer steps; experiments
may override that value independently.

The startup and final performance logs print the active iteration limit. For a
reproducible strict baseline run, still pass the settings explicitly:

```bash
./lbm_gpu \
  --mode gpu \
  --params params.in \
  --poisson split \
  --pressure-boundary split \
  --poisson-check-interval 100 \
  --poisson-tolerance 1e-3 \
  --poisson-iteration-limit 0
```
