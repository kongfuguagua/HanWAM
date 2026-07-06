# V3 temperature continuity fix

## Root cause

V2 directly used a `HistGradientBoostingRegressor` output as room temperature.
The one-hour thermal proxy feature crossed a root-tree threshold at step 824
(1.1444 h):

```text
feature[58]: 28.498564 -> 28.496485
tree threshold: 28.497917
```

The controls were unchanged in X12 and changed only one EEV count in X13, so
the 0.634/1.036 degC steps were nonphysical tree discontinuities.

## State equation

V3 treats the direct tree prediction as a target and integrates a causal
thermal-capacitance state:

```text
alpha = 1 - exp(-dt / 90 s)
T[k+1] = T[k] + clip(alpha * (T_direct[k+1] - T[k]), +/-0.5 degC/min)
```

The 90 s constant and 0.5 degC/min rate were selected using validation runs
only. The final test set was not used for tuning.

## Results

| Model | Test MAE | Test RMSE | Maximum simulated 5 s step |
|---|---:|---:|---:|
| V1 | 0.3547 degC | 0.4699 degC | unconstrained |
| V2 | 0.3275 degC | 0.4345 degC | 1.036 degC in X13 |
| V3 | 0.3422 degC | 0.4467 degC | **0.0417 degC** across all 48 runs |

For X12/X13, V3 MAE is 0.092/0.178 degC respectively. V2 remains archived
for reproducibility, while V3 is the physically consistent recommended model.

