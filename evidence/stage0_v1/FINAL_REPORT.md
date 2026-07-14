# Dual-residual Stage 0 v1 — final report

## Verdict

**Go**, limited to the strict unbounded GPU split+split dual-residual baseline
and its audit instrumentation in `/home/jzh/whileLBM_dualresidual_v1`.

The original `/home/jzh/whileLBM` was not modified.  No test artifact,
`cleanLBM1000`, 1000-step/8000-step run, or training was used.

This does **not** yet implement or validate the proposed `p+hneq` tail-state
warm start.  It establishes the correct strict convergence reference against
which that Oracle must be tested.

## Frozen semantics

- GPU `split+split` only.
- `poisson_iteration_limit=0` (unbounded), `check_interval=100`,
  `tolerance=1e-3`.
- Production map: collision with `tau_h=1/rho+1/2` and projected source,
  periodic pull stream, x-slip, then production `p=sum_i(h_i)`.
- Exit only when both the existing pressure relative-change residual and the
  gauge-aware kinetic fixed-point residual are `<1e-3`.
- Source mean, pressure gauge mean, pressure relative-change, and kinetic
  residual use fixed 256-thread block trees plus increasing-block-order host
  Kahan reduction.  This removes atomic ordering noise; it is not an
  acceleration mechanism.
- The shadow map never writes active `h`, `p`, or `p_prev`.

## Formula-to-code mapping

| Contract | Implementation |
|---|---|
| `C(h,p)` | `collisionPressureKernel` |
| periodic `S` | `streamKernel` |
| x-slip `B` | `slipBounceBackKernel` |
| `p=sum_i(h_i)` | `computePressureKernel` |
| `R_h=h-BS[C(h,p)]`, `R_p=p-sum_i(h_i)` | `fixedPointResidualBlockKernel` |
| dual exit and gauge order | `InamuroCUDA::doPressurePoisson` |
| real-state replay | `InamuroCUDA::runFixedPointStateAudit` and `oracle/verify_fixed_point_state_torch.py` |

`R_p` explicitly consumes the output of the production
`computePressureKernel`; it does not contain a duplicate private summation.

## Numerical gates

| Gate | Result | Limit |
|---|---:|---:|
| Synthetic C++ fixed-point term error | `1.24345e-14` | `1e-12` |
| Synthetic Torch CPU/CUDA worst error | `3.33067e-16` | `1e-12` |
| Real step1–2 C++ worst elementwise error | `3.33067e-16` | `1e-12` |
| Real step1–2 C++ fixed-term relative error | `1.44540e-13` | `1e-12` |
| Real step1–2 Torch CPU/CUDA worst gated error | `6.66134e-16` | `1e-12` |
| Real step1–2 C++/Torch direct component error | `5.55112e-17` | `1e-12` |
| Gauge full-map covariance | `<=6.66134e-16` | `1e-12` |
| Pressure JVP/VJP | finite and nonzero | required |
| NaN/Inf | `0` | `0` |

The real float64 checkpoints are captured after gauge and dual acceptance, but
before `correct_uvw` and `h_i=E_i p`; fields are exactly
`p,rho,u_x,v_y,w_z,h[15]`.

## Development step1–2

| Step | Iterations | Pressure residual | Fixed-point residual | Active-state bit mismatches |
|---:|---:|---:|---:|---:|
| 1 | 6600 | `9.391749e-4` | `9.100952e-6` | `0` |
| 2 | 10300 | `9.618957e-4` | `1.003034e-5` | `0` |

The two independent diagnostics CSVs are byte-identical.  Shadow auditing
checked `1,495,203,840` active `h` values and `99,680,256` active `p` values in
the two final step records, with zero mismatch.

## Twenty-step gates

- Strict dual baseline: `20/20` accepted, `0` fallback, total `130600`
  iterations, range `2400–11000` per step.
- Maximum final pressure residual: `9.9933069857e-4`.
- Maximum final kinetic fixed-point residual: `1.0725536289e-5`.
- Two full dual20 diagnostics files from independently built binaries are
  byte-identical.
- Route5 exact-pressure bypass requalification: `20/20` bitwise equal,
  `max_field_rel=0`, pressure iterations `0`.

The Route5 requalification is only a bypass-equivalence gate.  Its
`converged=1` at zero iterations is not evidence of dual-residual Poisson
convergence.

## Evidence

- Summary: `stage0_summary.json`
- Frozen input provenance: `../source_snapshot/SOURCE_PROVENANCE.md` and
  `../source_snapshot/SOURCE_SHA256SUMS`; the post-run recheck is
  `../source_snapshot/ORIGINAL_UNCHANGED_AUDIT.txt` (`26/26`, zero mismatch).
- Final build: `final_validation/build/`
- Synthetic audit: `final_validation/tbook/`
- Real step1–2 states and diagnostics: `final_validation/dev_step1_2/run_final/`
- C++ real-state audit: `final_validation/real_state_cpp/`
- Torch real-state audit: `final_validation/real_state_torch/results_final.json`
- Direct C++/Torch real-state cross-audit: `final_validation/real_state_cross/`
- Dual20: `final_validation/dual20_post_rebuild/`
- Route5 20-step requalification:
  `final_validation/oracle20_route5_post_rebuild/`
- Complete SHA-256 manifest: `FINAL_SHA256SUMS`

Optional `compute-sanitizer` memcheck could not attach under this WSL/WDDM
environment (`Device not supported`); this is recorded in
`final_validation/memcheck/`.  The audited executable itself still returned
`pass=1`.  Memcheck is not counted as a passed gate.

## Next route boundary

The next isolated change may implement the `p_tail+hneq_tail` Oracle capture
and reconstruction, but it must keep this solver, dual gate, unbounded loop,
and check interval frozen.  No learning claim is authorized by this Stage 0.
