# GPU split+split dual-residual contract

This isolated branch preserves the frozen `/home/jzh/whileLBM` GPU pressure
iteration and adds a separately switchable kinetic fixed-point diagnostic.

## Frozen production map

For every pressure iteration:

1. D3Q15 collision with `tau_h = 1/rho + 1/2` and the projected source.
2. Periodic pull stream in x/y/z.
3. Production x-slip overwrite.
4. `p = sum_q h_q`.

At each 100-iteration check, the existing synchronized pressure gauge is applied
first.  Shadow mode then evaluates one additional copy of the same map without
swapping pointers or writing active `p`, `h`, or `p_prev`.

Strict dual mode also replaces the source mean, pressure mean, and pressure
relative-change atomic reductions with a fixed 256-thread block tree followed
by increasing-block-order host Kahan summation.  This is a determinism control,
not an acceleration mechanism.

Let `g` be the frozen pressure-gauge target and `F(h,p)=BS[C(h,p)]`:

```text
R_h = h - F(h,p)
R_p = p - sum_q h_q
r_h = sum(abs(R_h)) /
      max(sum(abs(h-E*g)), sum(abs(F(h,p)-E*g)))
r_p = sum(abs(R_p)) /
      max(sum(abs(p-g)), sum(abs(sum_q(h)-g)))
r_fp = max(r_h, r_p)
```

A zero denominator maps to zero only when its numerator is also zero; otherwise
it is non-finite and the run fails.

## Modes

- Default: original pressure-relative-change exit; no shadow evaluation.
- `--poisson-fixed-point-shadow`: record `r_h`, `r_p`, and `r_fp`; do not alter exit.
- `--poisson-dual-residual`: enable shadow and require both
  `pressure_relative_change < 1e-3` and `r_fp < 1e-3`.
- `--poisson-shadow-state-audit`: bitwise-check active `h` and physical `p`
  before/after every shadow evaluation (development evidence only).
- `--poisson-fixed-point-dump-dir DIR`: dump the converged pre-`correct_uvw`,
  pre-`h_i=E_i p` float64 state for independent CPU/GPU/Torch replay.

Dual mode rejects any configuration other than GPU split+split,
`check_interval=100`, `tolerance=1e-3`, `poisson_iteration_limit=0`, unit
relaxation, deterministic reductions, no spatial diagnostic, and all optional
Poisson accelerators disabled.

The fixed-point checkpoint contains only the live converged state
`p,rho,u_x,v_y,w_z,h[15]`.  It is written after gauge and both gates pass, but
before the production `correct_uvw` and `update_hh`; it contains no future or
target-derived field.

The CPU full solver is not part of this unbounded production contract; it is
used only as an independent single-map reference.
