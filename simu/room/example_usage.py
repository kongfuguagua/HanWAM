"""V3 5秒在线环境最小示例。"""
try:
    from .simulator import ContinuousEnthalpyRoomEnv
except ImportError:  # direct script execution
    from simulator import ContinuousEnthalpyRoomEnv


env = ContinuousEnthalpyRoomEnv()
print(env.reset(T_out=35.0, T_in=30.0, T_out_coil=37.0, T_in_coil=24.0))

for _ in range(720):
    observation = env.step(freq=40.0, eev=180.0, fan_out=750.0)

print(observation)

