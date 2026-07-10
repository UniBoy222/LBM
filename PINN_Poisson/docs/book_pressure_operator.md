# Book-consistent pressure operator

## Sole source

The only formula and implementation source for this operator is:

`/home/jzh/.codex/attachments/3dca169c-0c51-4715-95dc-370f77226c5a/Multiphase Lattice Boltzmann Methods Theory and Application（拖移项目） 2.pdf`

The attachment is a 29-page extract. Printed book page 174 is attachment page 8;
printed pages 180-191 are attachment pages 14-25.

## One pressure iteration

The implementation follows the appendix order exactly:

1. `firstord` computes `du/dx + dv/dy + dw/dz` from all 14 moving D3Q15
   directions, divided by 10. X is symmetric; Y and Z are periodic.
2. `correction` applies Eqs. (6.42)-(6.43), with lattice `dx=1`.
3. `stream(hh)` performs periodic D3Q15 propagation.
4. `slip_bounceback(hh)` applies the appendix's x-wall direction copies.
5. `getp` applies Eq. (6.44): `p=sum_i(h_i)`.

The differentiable implementation is
`PINN_Poisson/scripts/book_pressure_operator.py`. Its fixed-point loss is
`mean((T_book(h_pred)-h_pred)^2)` and pressure is never predicted separately.

## Gate

Run:

```bash
make -C GPU test-book-pressure
```

The gate compares every stage for one deterministic step:

- independent CPU appendix implementation vs production CUDA kernels;
- Torch CPU vs CPU appendix implementation;
- Torch CPU vs production CUDA kernels;
- Torch CUDA vs CPU appendix implementation;
- CUDA autograd through the complete operator.

Training must not start unless all stages pass. Simple Laplacian residuals,
scalar/simple Poisson, guessed source-aware `hh`, and derived pressure closures
are not valid physics targets for this direction.
