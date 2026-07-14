# API Servicer

本目录提供 HTTP JSON 控制服务。服务启动时读取 YAML 配置，实例化 `hanwam_wm_mpc` controller adapter，并按 `unit_id` 维护独立的 MPPI warm-start 和短周期缓存动作。

## 启动

```bash
python -m servicer.api_servicer.main \
  --config servicer/config/api_service_hanwam.yml
```

宿主机配置使用相对路径，默认指向 E065-L10：

```text
servicer/config/api_service_hanwam.yml
```

容器配置使用 `/app/...` 绝对路径：

```text
servicer/config/api_service_hanwam_docker.yml
```

配置字段包括：

```text
service.host / service.port / service.num_threads / service.device
logging.level / logging.mode / logging.path / logging.payloads
controller.type / controller.algorithm_config / controller.checkpoint
```

checkpoint 不提交到仓库，启动真实 HanWAM adapter 前需要外部挂载。

## API

```text
GET  /healthz
GET  /readyz
GET  /v1/metadata
POST /v1/plan
POST /v1/reset
```

`/v1/plan` 主要字段：

```text
controller_type = hanwam_wm_mpc
unit_id = ac_xxx
mode = 1
step_seconds = 5
target_temperature_c = 27.0
elapsed_seconds = 300.0
obs_history = oldest -> newest
act_history = oldest -> newest
deadline_seconds = optional
return_debug = optional
```

HanWAM adapter 会用第一次有效请求的 `obs_history[0].T_in`、`obs_history[0].T_out` 和 `target_temperature_c` 自动计算 DDL，并缓存到对应 `unit_id`。目标温度变化时会重新计算 DDL，同时清空该设备的 planner warm-start 和 cached actions。

`return_debug=true` 时会回显原始请求、校验后请求、请求体 SHA-256 和字节数。请求内容可能包含设备运行数据，联调结束后应关闭。

## Docker

```bash
make -C servicer docker-build IMAGE_TAG=cpu
make -C servicer docker-run IMAGE_TAG=cpu OUTPUT_DIR=outputs/servicer
curl http://127.0.0.1:24243/readyz
```

登录镜像仓库使用环境变量，不要把密码写入文件：

```bash
export REGISTRY_USERNAME='your-user'
export REGISTRY_PASSWORD='******'
make -C servicer docker-login
```

## 测试

```bash
python -m compileall servicer control/HanWAM control/MiniController
python -m unittest discover -s servicer/tests
```
