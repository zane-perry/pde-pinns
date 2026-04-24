# Statistical uncertainty analysis

- Seeds: [0, 1, 2, 3, 4]
- N_DATA: 0
- Baseline included: No
- Generated from existing saved outputs only; no models were retrained.

## Main interpretation

This analysis treats the trained informed PINN as a random outcome of initialization, collocation sampling, and optimization. Final metrics are therefore summarized empirically across repeated seeds rather than reported only as single-run point estimates.

## Elliptic

- **rel_l2_test**: mean = 1.433955e-03, std = 6.257905e-04, median = 1.513688e-03, 95% bootstrap CI for mean = [9.551099e-04, 1.932402e-03]
- **heldout_residual_mse**: mean = 3.749992e-03, std = 3.970934e-03, median = 1.984464e-03, 95% bootstrap CI for mean = [1.010025e-03, 7.196507e-03]
- **boundary_rel_l2**: mean = 1.234970e-07, std = 0.000000e+00, median = 1.234970e-07, 95% bootstrap CI for mean = [1.234970e-07, 1.234970e-07]
- **h1_semi_rel**: mean = 2.161957e-03, std = 1.031408e-03, median = 2.069343e-03, 95% bootstrap CI for mean = [1.359832e-03, 2.964082e-03]
- **mean_barrier_violation**: mean = 0.000000e+00, std = 0.000000e+00, median = 0.000000e+00, 95% bootstrap CI for mean = [0.000000e+00, 0.000000e+00]
- **max_lower_violation**: mean = 0.000000e+00, std = 0.000000e+00, median = 0.000000e+00, 95% bootstrap CI for mean = [0.000000e+00, 0.000000e+00]

- Relative L2 coefficient of variation: 4.364e-01

## Parabolic

- **rel_l2_test**: mean = 2.696126e-04, std = 1.298986e-04, median = 2.311163e-04, 95% bootstrap CI for mean = [1.825447e-04, 3.820331e-04]
- **final_time_rel_l2**: mean = 2.725698e-04, std = 1.058843e-04, median = 2.497023e-04, 95% bootstrap CI for mean = [1.904929e-04, 3.544908e-04]
- **heldout_residual_mse**: mean = 1.779899e-05, std = 2.032916e-05, median = 9.031361e-06, 95% bootstrap CI for mean = [8.212012e-06, 3.605293e-05]
- **mean_barrier_violation**: mean = 5.852965e-10, std = 0.000000e+00, median = 5.852965e-10, 95% bootstrap CI for mean = [5.852965e-10, 5.852965e-10]
- **frac_negative**: mean = 4.149378e-03, std = 0.000000e+00, median = 4.149378e-03, 95% bootstrap CI for mean = [4.149378e-03, 4.149378e-03]
- **neg_part_l2**: mean = 5.631396e-09, std = 0.000000e+00, median = 5.631396e-09, 95% bootstrap CI for mean = [5.631396e-09, 5.631396e-09]

- Relative L2 coefficient of variation: 4.818e-01

## Hyperbolic

- **rel_l1_test**: mean = 1.655171e-01, std = 1.899615e-02, median = 1.700498e-01, 95% bootstrap CI for mean = [1.494655e-01, 1.793672e-01]
- **final_time_rel_l1**: mean = 1.050871e-01, std = 3.077047e-02, median = 1.223560e-01, 95% bootstrap CI for mean = [8.173611e-02, 1.284382e-01]
- **final_time_rel_l2**: mean = 1.233572e-01, std = 2.835934e-02, median = 1.361245e-01, 95% bootstrap CI for mean = [1.012960e-01, 1.449391e-01]
- **heldout_residual_mse**: mean = 2.306235e-01, std = 2.366679e-02, median = 2.198384e-01, 95% bootstrap CI for mean = [2.172758e-01, 2.515824e-01]
- **heldout_entropy_violation**: mean = 4.112125e-02, std = 7.197366e-03, median = 4.167104e-02, 95% bootstrap CI for mean = [3.582149e-02, 4.688030e-02]
- **shock_location_error_final**: mean = 1.500000e-02, std = 1.541104e-02, median = 1.250000e-02, 95% bootstrap CI for mean = [4.500000e-03, 2.800000e-02]

- Relative L2 coefficient of variation: 5.343e-02

- Relative L1 coefficient of variation: 1.148e-01

## Notes

- Wider uncertainty bands or larger seed-to-seed spread suggest greater sensitivity to initialization and collocation randomness.
- This is an empirical repeated-training uncertainty analysis, not a full Bayesian posterior analysis.
- The summary statistics here can directly support the report's reliability / statistical perspectives section.
