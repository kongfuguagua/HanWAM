"""5秒在线环境最小示例。"""
try:
    from .simulator import OnlineEnthalpyRoomEnv
except ImportError:  # direct script execution
    from simulator import OnlineEnthalpyRoomEnv

env=OnlineEnthalpyRoomEnv()
print(env.reset(T_out=35.0,T_in=30.0,T_out_coil=37.0,T_in_coil=28.0))

# 控制系统每5秒调用一次；不需要预先给出未来控制。
for _ in range(720):
    observation=env.step(freq=40.0,eev=180.0,fan_out=750.0)
    # observation["T_in"] 即当前5秒时刻的室内环境温度。
print(observation)
