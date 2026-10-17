# Model card: NeutroNet DeepONet surrogate

## Intended use

The model approximates the mapping from a distributed neutron source and heterogeneous material cross sections to scalar flux in a one-dimensional slab. It is intended for operator-learning research, reproducible numerical experiments, and surrogate-model demonstrations.

It is not intended for reactor design, licensing, criticality safety, clinical use, or decisions requiring a validated high-fidelity transport calculation.

## Architecture

The input contains three 100-value fields: source `Q`, total cross section `Sigma_t`, and scattering cross section `Sigma_s`. Each field passes through its own `100 -> 128 -> 64` encoder. The three 64-value representations are concatenated and passed through a `192 -> 192 -> 64` fusion network. The coordinate trunk is `1 -> 128 -> 128 -> 64`. ReLU activations are used between linear layers. A dot product between the fused branch representation and trunk representation, plus a learned scalar bias, produces the flux at each coordinate.

The checkpoint has 137,985 trainable parameters and is stored at `results/final_model/model.pt`.

## Training

- Dataset: `data/final_model/train.npz`
- Training cases: 2,800, balanced across seven source families
- Material topology: one or two layers
- Spatial grid: 100 cells, 10 cm length
- Reference solver: S32 discrete ordinates, diamond difference, vacuum boundaries
- Objective: batch mean of squared per-sample relative L2 error
- Optimizer: AdamW
- Initial learning rate: 3e-4
- Weight decay: 1e-6
- Batch size: 32
- Scheduler: ReduceLROnPlateau, factor 0.5
- Maximum epochs: 1,000
- Early-stopping patience: 120
- Seed: 20260920
- Selected epoch: 651
- Completed epochs: 771
- Training hardware: NVIDIA GeForce GTX 1650 with Max-Q Design
- Recorded training time: 167.53 seconds

Normalization statistics were fit only on the training split. Model selection used only the validation split.

## Evaluation

The following complementary metrics are computed over all scalar-flux values in each split. MSE is in squared-flux units; RMSE and MAE are in flux units. NRMSE is RMSE divided by target RMS. Relative L1 and relative L2 are first calculated per case and then averaged.

| Split | MSE | RMSE | MAE | NRMSE | Mean relative L1 | Mean relative L2 | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 2.5797 | 1.6061 | 0.6040 | 7.75% | 3.62% | 4.09% | 0.9888 |
| In distribution | 5.2952 | 2.3011 | 0.5920 | 12.01% | 3.64% | 4.12% | 0.9725 |
| Three-layer OOD | 13.2486 | 3.6399 | 1.9906 | 22.91% | 16.45% | 19.25% | 0.8818 |
| Unseen-source OOD | 1.3965 | 1.1817 | 0.4773 | 6.56% | 3.33% | 3.75% | 0.9914 |

Absolute-error and interface diagnostics are also retained:

| Split | Median absolute | Maximum absolute | Mean bias | Interface MAE | Interface RMSE |
|---|---:|---:|---:|---:|---:|
| Validation | 0.2050 | 18.9972 | -0.1680 | 0.9017 | 1.9362 |
| In distribution | 0.2002 | 48.9476 | -0.1794 | 1.1314 | 4.0507 |
| Three-layer OOD | 0.8622 | 31.4760 | -0.9047 | 2.2557 | 3.6683 |
| Unseen-source OOD | 0.1797 | 20.4981 | -0.1682 | 0.6864 | 1.2947 |

For completeness, the distribution of per-case relative L2 errors is:

| Split | Mean relative L2 | 95% bootstrap CI | Median | P95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Validation | 4.09% | 3.77-4.43% | 3.16% | 10.08% | 23.58% |
| In distribution | 4.12% | 3.80-4.47% | 3.25% | 9.42% | 30.06% |
| Three-layer OOD | 19.25% | 18.08-20.47% | 16.59% | 41.36% | 91.82% |
| Unseen-source OOD | 3.75% | 3.53-3.98% | 2.91% | 9.19% | 21.64% |

The bootstrap intervals summarize test-case variability, not variability across independently trained networks or independently generated datasets. Pointwise MAPE is not reported because scalar flux can approach zero near vacuum boundaries, making that statistic unstable and potentially misleading.

## Limitations

- Three-layer material extrapolation is weak and has a large worst-case error.
- Results come from one final network seed and one generated split.
- Training cross sections are synthetic rather than benchmark-derived.
- The internal solver is tested but not cross-validated here against an independent transport implementation.
- Performance is established only for the stated 1D geometry, discretization, parameter ranges, and vacuum boundaries.
- GPU timing should not be compared with end-to-end deterministic-solver time without a controlled benchmark.
