# HanWAM AC Energy-Saving Control

This repository is a clean code snapshot for air-conditioner energy-saving
control. It contains the HanWAM controller, simulation modules, API servicer,
configuration, and mock-friendly tests. Real experiment data, trained weights,
and generated outputs are intentionally kept outside the repository.

## Layout

```text
control/    HanWAM, PID, and MiniController experiment framework
simu/       actuator, frequency, energy, and room simulation modules
servicer/   HTTP JSON service wrapper for online HanWAM planning
data/       data-loading code and external data conventions
tests/      unit and interface tests using temporary mock assets
```

## Current HanWAM Entry Points

- Config: `control/HanWAM/config/hanwam.yml`
- Training: `python -m control.HanWAM.train --config control/HanWAM/config/hanwam.yml --stage both`
- Evaluation: `python -m control.MiniController.main --config control/HanWAM/config/hanwam.yml --stage eval`
- Servicer config: `servicer/config/api_service_hanwam.yml`

The configured checkpoint path is
`control/HanWAM/checkpoints/hanwam_mode1.pt`. The path is preserved for local
runs, but the `.pt` file itself is not committed.

## Clean Snapshot Rules

The repository should not contain real `csv`, `pt`, `pth`, `pkl`, `joblib`,
spreadsheet, PDF, image, log, or generated output artifacts. Tests that need
data or models create temporary mock files at runtime.

Use `data/README.md` for the external data convention and
`control/HanWAM/README.md` for the HanWAM architecture and training notes.
