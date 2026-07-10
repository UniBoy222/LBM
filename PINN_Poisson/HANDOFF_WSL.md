# WSL Handoff - 2026-07-11

## Active direction

- Sole formula source: the supplied PDF, book Eq. (6.42)-(6.44) and appendix Fortran.
- Network output is complete `h_i[15]`; pressure is strictly `sum_i(h_i)`.
- Physics loss is the exact `correction -> stream -> slip_bounceback -> getp` operator.
- Simple/scalar Poisson and guessed source-aware `hh` are disabled.

## Critical finding

The pressure loop uses gradients computed in `collision()` before `stream/getuv`.
Therefore a snapshot of the updated `u*` cannot reproduce the actual pressure source.
Feature format v2 (`PINNF2`) adds the exact seventh field:

`div_u_source = upx + vpy + wpz`

All earlier six-channel `h_i` models and cross-run pre/post manifests are invalid for
strict continuation. Do not resume from them.

## Valid data and gates

- Exact same-trajectory base data:
  `PINN_Poisson/data/h_exact_base_v2_120/`
- Base manifest/stats:
  `PINN_Poisson/data/h_exact_base_v2_manifest.csv`
  `PINN_Poisson/data/h_exact_base_v2_stats.json`
- Split: steps 1-80 train, 81-100 val, 101-120 test.
- Gate: 120/120 pass; max `|p-sum(h)|=2.22e-16`; max post pressure mismatch `2.97e-8`.
- Replay sanity: exact step 19 converges at iteration 100 with residual `9.20719e-4`.

## Candidate diagnosis and hard replay

- Per-candidate diagnostics:
  `PINN_Poisson/results/benchmarks/h15_p50_step200_candidate_diagnostics.csv`
- Hard-step list:
  `PINN_Poisson/data/h15_p50_hard_steps.txt`
- 120 candidates, 97 hard states:
  fallback 36, pressure peak 65, false accept 50, negative direct gain 39,
  negative future-20 gain 19.
- Captured exact trajectory pre states/features:
  `PINN_Poisson/data/h15_p50_hard_pre_capture_v2/`
  `PINN_Poisson/data/multistep_closed_loop/h15_p50_hard_pre_capture_v2/`
- Full-Poisson replay targets and raw manifest:
  `PINN_Poisson/data/h15_p50_hard_replay_v2/`
- At max 5000 book iterations, 37/97 states converge; 60 fail the `1e-3`
  residual gate. Failed residual min/median/max: `1.057e-3 / 1.320e-3 / 1.058e-2`.
- Do not add failed targets to training. Training is paused at this gate.

## Interrupted experiment

Steps 3 and 29 were retried with 10000 iterations, but the client interrupted before
their stdout was captured. No replay process is currently active. Re-run these two
representatives first and record residuals before deciding whether longer exact-book
iterations are viable.

## Last valid model result (historical only)

The six-channel p50 model reached 200-step `381.75 iter/step`, 84 accepts/36
fallbacks, pressure final `1.14e-2`, checkpoint max `5.29e-2`, rho max `4.86e-4`,
fei max `4.66e-4`. It is invalid for further training because it lacks
`div_u_source`, but its closed-loop trajectory is the source of the hard cases.

## Acceptance gates

After rebuilding the seven-channel dataset and retraining:

- pressure checkpoint max `<=2.1e-2`
- fallback rate `<5%`
- rho/fei no worse than the current 200-step result
- average Poisson reduction at least `50 iter/step`
- only then run 1000-step, then 8000-step

## Resume order

1. Verify all files listed above exist and hashes/counts match the source machine.
2. Re-run hard replay steps 3 and 29 at 10000 iterations and record convergence.
3. Resolve the 60 failed exact-book replay states without changing formulas.
4. Build a combined manifest preserving the original 120 samples and valid hard replay
   samples; split by trajectory and contiguous time blocks to prevent leakage.
5. Re-run manifest consistency and fixed-point gates.
6. On RTX 5080: CUDA check, 1-epoch smoke, single-sample overfit, formal training.
7. Sync model/logs back, run local 200-step gate; do not skip directly to long runs.

Training rules are in the repository root `AGENTS.md`. Remote project root must be
`/home/jzh/PINN/LBM`; training uses `PINN_Poisson/.venv` and tmux.
