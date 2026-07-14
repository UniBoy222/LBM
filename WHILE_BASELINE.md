# GPU while baseline contract

`GPU/lbm_gpu` defaults to the strict pressure-solver contract below:

- split Poisson and split pressure boundary;
- `while (true)` iteration until convergence;
- residual check interval: `100`;
- convergence tolerance: `1e-3` with strict `residual < tolerance` acceptance;
- `poisson_iteration_limit=0`, meaning unbounded;
- immediate failure on non-finite source, pressure, gauge, or residual values;
- explicit overflow checks for the per-step Poisson iteration counter, CUDA-graph
  segment multiplication, accumulated Poisson iterations, and time-step counter.

The startup and final performance logs print the active iteration limit. For a
reproducible strict baseline run, still pass the settings explicitly:

```bash
./lbm_gpu \
  --mode gpu \
  --params pinn_restart_assets/configs/params_baseline_8000_stride1.in \
  --poisson split \
  --pressure-boundary split \
  --poisson-check-interval 100 \
  --poisson-tolerance 1e-3 \
  --poisson-iteration-limit 0
```
