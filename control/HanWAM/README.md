# HanWAM

HanWAM is the current world-model MPC control line. It uses a block latent
world model, a future-latent parameter prober, a single-lag thermal mechanism,
and MPPI planning.

## Active Files

```text
config:          control/HanWAM/config/hanwam.yml
checkpoint path: control/HanWAM/checkpoints/hanwam_mode1.pt
tests:           tests/test_hanwam_block.py
```

Checkpoint architecture version is `hanwam_block_v1`. Trained `.pt` artifacts
are intentionally excluded from this repository; place local weights under
`control/HanWAM/checkpoints/` before running evaluation or the servicer.

## Architecture

One block contains 12 five-second frames, or 60 seconds. The current config uses
4 history blocks and 8 future blocks.

```text
history[48 x (obs, action)]
        |
        +-- encode once --> z0
                              |
candidate action paths --------+--> action-conditioned latent rollout
                                      |
                                future latents
                                      |
                        bounded parameter prober
                                      |
                    single-lag thermal mechanism
                                      |
                       future T_in and energy
                                      |
                 MPPI reach/clamp/energy/action cost
                                      |
                       first 5s command
```

Two implementation properties are intentional:

- future latents drive the physical prober and cannot be bypassed;
- each MPPI replan encodes history once, then reuses that prepared context for
  all candidates and nominal evaluation.

## Training And Evaluation

```bash
python -m control.HanWAM.train \
  --config control/HanWAM/config/hanwam.yml \
  --stage both --device cuda

python -m control.MiniController.main \
  --config control/HanWAM/config/hanwam.yml \
  --stage eval
```

Stage I trains action-conditioned latent dynamics with Epps-Pulley SIGReg on
predicted rollout latents. Stage II freezes the world model and trains the
parameter prober against block-level `T_in_delta` and `electric_kwh_delta`
targets.

## Online Control Constraints

- compressor candidate frequency: 10-80 Hz, no shutdown action;
- EEV: 100-270;
- outdoor fan: fixed at 750 rpm;
- planner replans every 10 seconds and emits 5-second actions;
- cost parts: reach, clamp, energy, and action.

## Diagnostics

```text
action_sensitivity.py     checks future-action influence on latent rollout
action_response_sweep.py  checks checkpoint response across candidate actions
action_analysis.py        summarizes closed-loop action behavior
visualize_rollouts.py     plots retained rollout outputs when present locally
```

Tests use generated mock data and light models where possible, so the clean
repository can be validated without committing real data or checkpoints.
