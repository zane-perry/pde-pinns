# Statistical uncertainty analysis

- Seeds: [0, 1, 2, 3, 4]
- N_DATA: 0
- Baseline included: No

## Main interpretation

This analysis treats the trained informed PINN as a random outcome of initialization, collocation sampling, and optimization. Final metrics are therefore summarized empirically across repeated seeds rather than reported only as single-run point estimates.

## Elliptic

- **rel_l2_test**: mean = 8.378203e-03, std = 7.922825e-03, median = 8.013309e-03, 95% bootstrap CI for mean = [2.736212e-03, 1.480119e-02]
- **heldout_residual_mse**: mean = 1.342523e-01, std = 1.696959e-01, median = 5.281634e-02, 95% bootstrap CI for mean = [3.860866e-02, 2.887304e-01]
- **h1_semi_rel**: mean = 1.170582e-02, std = 9.614205e-03, median = 9.492800e-03, 95% bootstrap CI for mean = [5.456187e-03, 2.013577e-02]
- **min_pred**: mean = 0.000000e+00, std = 0.000000e+00, median = 0.000000e+00, 95% bootstrap CI for mean = [0.000000e+00, 0.000000e+00]
- **max_pred**: mean = 1.001488e+00, std = 5.871088e-03, median = 9.990239e-01, 95% bootstrap CI for mean = [9.970475e-01, 1.005928e+00]

- Relative L2 coefficient of variation: 9.456e-01

## Parabolic

- **rel_l2_test**: mean = 1.731385e-04, std = 2.879673e-04, median = 1.141900e-05, 95% bootstrap CI for mean = [3.905850e-06, 4.383934e-04]
- **final_time_rel_l2**: mean = 4.880320e-04, std = 8.662432e-04, median = 2.746203e-05, 95% bootstrap CI for mean = [8.628782e-06, 1.284494e-03]
- **heldout_residual_mse**: mean = 1.418402e-05, std = 2.107965e-05, median = 3.965984e-08, 95% bootstrap CI for mean = [1.203283e-08, 3.301473e-05]
- **mean_barrier_violation**: mean = 4.519422e-10, std = 0.000000e+00, median = 4.519422e-10, 95% bootstrap CI for mean = [4.519422e-10, 4.519422e-10]
- **frac_negative**: mean = 4.149378e-03, std = 0.000000e+00, median = 4.149378e-03, 95% bootstrap CI for mean = [4.149378e-03, 4.149378e-03]
- **neg_part_l2**: mean = 3.705652e-09, std = 0.000000e+00, median = 3.705652e-09, 95% bootstrap CI for mean = [3.705652e-09, 3.705652e-09]

- Relative L2 coefficient of variation: 1.663e+00

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
