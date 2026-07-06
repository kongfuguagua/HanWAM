# HanWAM Architecture

HanWAM is the active WAM implementation for HVAC MPC.

## Schema

Observation:

```text
T_in, T_set, mode, freq_target, freq, eev, fan_out,
elapsed_seconds, T_out, energy_cum
```

Action:

```text
freq_target, eev, fan_out
```

Prober output:

```text
T_in, freq, T_in_delta, electric_kwh_delta
```

## Training

The model is trained and evaluated with a history window. The current
validation config uses:

```text
history_steps = 12
horizon_steps = 24
```

Stage 1 trains only the latent world model:

```text
state history encoder + action history encoder + action encoder + GRU predictor
loss = multi-step latent MSE + SIGReg
```

The physical prober is frozen during Stage 1. No physical MSE, cumulative
temperature loss, slice-weighted loss, target encoder, EMA, direct policy, or
controller fallback is used.

Stage 2 freezes the latent world model and trains only the prober:

```text
stop-gradient latent rollout -> prober -> physical MPC variables
loss = physical prober MSE
```

## MPC

The active MPC objective is intentionally small:

```text
J = tracking + energy + action_smooth
```

Tracking follows a deadline-linear reference temperature trajectory. Energy is
the predicted accumulated `electric_kwh_delta`. Action smoothness penalizes
normalized action changes, including the first planned action relative to the
current actuator state.

The compressor deadband is handled as actuator-domain parameterization:

```text
0 < freq_target < 15Hz -> 0Hz
```

This prevents invalid low-frequency compressor commands without adding a
separate safety policy or fallback controller.

## Entrypoints

Full training:

```bash
CONFIG=control/HanWAM/config/hanwam.yml scripts/train_hanwam.sh
```

Smoke training:

```bash
CONFIG=control/HanWAM/config/hanwam_smoke.yml SWANLAB=0 scripts/train_hanwam.sh
```

Evaluation:

```bash
python -m control.HanWAM.eval --config control/HanWAM/config/hanwam_smoke.yml
```

## Validation Notes

`hanwam_design_validation.yml` is the current short design-validation run. It
uses the strict two-stage HanWAM design, 8192 train sequences, 4096 validation
sequences, `history_steps=12`, `horizon_steps=24`, and a 30s MPC execution
interval (`control_interval_steps=6`).

The first single-frame validation failed physically: MPC selected all-off
actions, consumed 0 kWh, and the room warmed up. A direct off/base/high latent
rollout diagnostic showed that the world model was nearly action-insensitive.
The accepted wiring fix is to feed state and action history windows to both
Stage 1 and Stage 2 training, and to the online planner.

On the first eligible test window (`X1_status_data_20260519103131`) after the
history-window fix:

```text
HanWAM one_four_hour: success=true, reach_time_s=11340,
final_error_c=0.1425, electric_kwh=1.1455
pid same window:      success=true, electric_kwh=2.2102
fixed same window:    success=true, electric_kwh=2.2190
historical window:    success=true, electric_kwh=4.6325
```

Action review:

```text
deadband_count=0
mean_abs_delta_freq_target=0.1247 Hz/step
planner_calls=480 over 2880 simulator steps
off_power_zero_cooling_steps=1 out of 1964 off/power-zero steps
```

Conclusion: the strict two-stage HanWAM design is now connected end to end and
can beat pid/fixed/historical on one 4h window, but it is not yet final SOTA.
The current model still has weak open-loop action sensitivity and the planner
uses high fan during many off steps. Next iterations should improve Stage 1
stability/action sensitivity and add a simple deployable fan/action cost inside
the MPC objective, without reintroducing direct policy, EMA, target encoder, or
fallback safety profiles.
