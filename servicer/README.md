# HanWAM API servicer

This directory provides an HTTP JSON wrapper for online control. The current
controller adapter is `hanwam_wm_mpc`, which loads the HanWAM config and a local
checkpoint, then serves `/v1/plan` requests.

## Local Run

```bash
python -m servicer.api_servicer.main \
  --config servicer/config/api_service_hanwam.yml
```

The host config uses repository-relative paths:

```text
algorithm_config: control/HanWAM/config/hanwam.yml
checkpoint:       control/HanWAM/checkpoints/hanwam_mode1.pt
log path:         outputs/servicer/hanwam_service.log
```

The checkpoint file is not committed. Add a trained local checkpoint before
starting a real HanWAM service.

## API

```text
GET  /healthz
GET  /readyz
GET  /v1/metadata
POST /v1/plan
POST /v1/reset
```

`/v1/plan` accepts:

```text
controller_type = hanwam_wm_mpc
unit_id = ac_xxx
mode = 1
step_seconds = 5
target_temperature_c = 27.0
elapsed_seconds = current task age in seconds
obs_history = recent frames, oldest to newest
act_history = recent actions, oldest to newest
return_debug = true or false
```

The exact history length is reported by `/v1/metadata`; the current HanWAM
config uses 48 observation/action frames.

## Docker

```bash
make -C servicer docker-build IMAGE_TAG=cpu
make -C servicer docker-run IMAGE_TAG=cpu
```

To push an image, provide registry settings at invocation time instead of
committing credentials:

```bash
export REGISTRY_PASSWORD='******'
make -C servicer docker-release \
  REGISTRY_HOST=registry.example.com \
  REGISTRY_USER=my-user \
  IMAGE_REPO=registry.example.com/team/hanwam-service \
  IMAGE_TAG=cpu
```

## Testing

```bash
python -m compileall servicer control/HanWAM control/MiniController
python -m unittest discover -s servicer/tests
make -C servicer validate-config
```

Adapter tests skip real-checkpoint execution when the external checkpoint is
absent. API contract tests use a fake controller and do not need model weights.
