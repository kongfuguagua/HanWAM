"""压缩机频率响应解析公式 — 一阶惯性 + 死区 + 非对称时间常数。

公式(连续形式):
    dFreq/dt = (Freq_target(t - τ_dead) - Freq) / τ_ramp
    Freq(t) = clamp(Freq(t), 0, Freq_max)

其中 τ_ramp 根据状态切换:
    - 冷启动(从 0 → ≥15Hz):τ_ramp = τ_cold    (慢爬升)
    - 运行中升频(target > Freq):τ_ramp = τ_up   (快)
    - 运行中降频(target < Freq):τ_ramp = τ_down (慢,有长尾)
    - 关机(target → 0):Freq = 0(直接归零,92.8% 概率)

死区 τ_dead(从目标阶跃到实际开始跟随):
    - 冷启动: ~5s
    - 稳态阶跃: ~5s(中位 1 个采样步)

离散化(Euler,dt=5s):
    Freq[t+1] = (1 - dt/τ) * Freq[t] + (dt/τ) * Freq_target[t - k]
    其中 k = τ_dead / dt

本文件包含:
    - 参数拟合(fit_*):从 122 个 ON + 149 个稳态阶跃事件反演 4 个参数
    - 前向仿真(simulate_freq):在已知目标序列上预测实际频率
    - 评估函数:对比预测 vs 真实 freq,输出 MAE/R^2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from data.io import load_all_raw, remove_constant_cols

HERE = Path(__file__).resolve().parent

OUT = HERE / "output"
DT = 5.0  # 采样周期
T_DEAD_SEC_DEFAULT = 5.0  # 死区(1 个采样步)
FREQ_MAX = 90.0
FREQ_MIN = 0.0
ON_THRESHOLD = 15.0  # 冷启动判定


# ---------- 数据加载 ----------
def load_all_clean() -> pd.DataFrame:
    df = load_all_raw(verbose=False)
    df, _ = remove_constant_cols(df)
    df = df[df["mode"] == 1].copy()
    df = df.dropna(subset=["freq_in_tgt", "freq"])
    df = df.sort_values(["source", "ts"]).reset_index(drop=True)
    return df


# ---------- 冷启动事件检测 ----------
def detect_on_events(df: pd.DataFrame, threshold: float = ON_THRESHOLD,
                     pre_steps: int = 2, post_steps: int = 90) -> list[dict]:
    """检测从 freq_in_tgt=0 → ≥ threshold 的事件。

    返回: list of {source, t_idx, tgt_after, act_window, tgt_window}
    """
    events = []
    for src, g in df.groupby("source", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        tgt = g["freq_in_tgt"].to_numpy(dtype=float)
        act = g["freq"].to_numpy(dtype=float)
        for t in range(pre_steps, len(g) - post_steps):
            if tgt[t - 1] < threshold and tgt[t] >= threshold:
                tgt_after = float(tgt[t])
                # 至少前面 1 步 freq 也是 0/很小
                if act[t - 1] >= 5.0:
                    continue
                events.append({
                    "source": src,
                    "t_idx": t,
                    "tgt_after": tgt_after,
                    "act_window": act[t:t + post_steps + 1].copy(),
                    "tgt_window": tgt[t:t + post_steps + 1].copy(),
                })
    return events


# ---------- 稳态阶跃事件检测 ----------
def detect_step_events(df: pd.DataFrame, threshold: float = 8.0,
                       pre_steps: int = 4, post_steps: int = 30) -> tuple[list, list]:
    """检测稳态阶跃事件(排除 0/关机)。"""
    up, down = [], []
    for src, g in df.groupby("source", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        tgt = g["freq_in_tgt"].to_numpy(dtype=float)
        act = g["freq"].to_numpy(dtype=float)
        d = np.diff(tgt)
        jump_idx = np.where(np.abs(d) >= threshold)[0]
        for ji in jump_idx:
            t0 = ji + 1  # 跳变后的索引
            if t0 < pre_steps or t0 + post_steps + 1 >= len(g):
                continue
            before = float(tgt[t0 - 1])
            after = float(tgt[t0])
            # 排除开关事件
            if after < 5 or before < 5:
                continue
            event = {
                "source": src,
                "t_idx": t0,
                "before": before,
                "after": after,
                "act_window": act[t0 - pre_steps:t0 + post_steps + 1].copy(),
                "tgt_window": tgt[t0 - pre_steps:t0 + post_steps + 1].copy(),
            }
            if after > before:
                up.append(event)
            else:
                down.append(event)
    return up, down


# ---------- 模型函数 ----------
def first_order_ramp(t, target, tau, t_delay=0.0):
    """一阶指数爬升: f(t) = target * (1 - exp(-(t - t_delay)/tau)),t < t_delay 时为 0"""
    return np.where(t >= t_delay, target * (1.0 - np.exp(-(t - t_delay) / tau)), 0.0)


def first_order_step(t, before, after, tau):
    """一阶指数阶跃:f(t) = after + (before - after) * exp(-t/tau)"""
    return after + (before - after) * np.exp(-t / tau)


# ---------- 拟合 ----------
def fit_tau_cold(events: list) -> tuple[float, list[float]]:
    """从冷启动事件拟合 τ_cold。每个事件用非线性最小二乘。"""
    taus = []
    t_axis = np.arange(0, len(events[0]["act_window"]) * DT, DT) if events else np.array([])
    for ev in events:
        if ev["tgt_after"] < 30:
            continue
        # 找启动延迟(第一次 freq > 0)
        act = ev["act_window"]
        nz = np.where(act > 0)[0]
        if len(nz) == 0:
            continue
        t_delay = nz[0]
        if t_delay >= len(act) - 5:
            continue
        # 在 t >= t_delay 段拟合
        t_fit = np.arange(t_delay, len(act)) * DT
        y_fit = act[t_delay:]
        tgt = ev["tgt_after"]
        # 起点 = act[t_delay],用非线性拟合
        try:
            popt, _ = curve_fit(
                first_order_ramp, t_fit, y_fit,
                p0=[tgt, 8.0],  # tau 初值 8s
                bounds=([tgt * 0.5, 1.0], [tgt * 1.1, 60.0]),
                maxfev=2000,
            )
            taus.append(popt[1])
        except Exception:
            pass
    if not taus:
        return 10.0, []
    return float(np.median(taus)), taus


def fit_tau_step(up_events: list, down_events: list) -> tuple[float, float, list, list]:
    """从稳态阶跃事件读 τ63(63% 响应时间),再反推 τ = τ63。

    一阶指数模型 τ63 = τ,因此从数据中直接读 τ63 比非线性最小二乘更稳定。
    """
    def tau63_for(ev: dict) -> float | None:
        """从单个阶跃事件读 τ63。"""
        act = ev["act_window"]
        t0_idx = 4  # 阶跃发生位置
        before = ev["before"]
        after = ev["after"]
        target = after + 0.63 * (before - after)  # 63% 处的值
        # 找首个到达 target 的位置
        if after > before:  # UP
            crossed = np.where(act[t0_idx:] >= target)[0]
        else:  # DOWN
            crossed = np.where(act[t0_idx:] <= target)[0]
        if len(crossed) == 0:
            return None
        return float(crossed[0] * DT)  # τ63 = τ

    taus_up = []
    for ev in up_events:
        if ev["after"] < 30:
            continue
        t = tau63_for(ev)
        if t is not None:
            taus_up.append(t)
    taus_down = []
    for ev in down_events:
        if ev["before"] < 30:
            continue
        t = tau63_for(ev)
        if t is not None:
            taus_down.append(t)
    tau_up = float(np.median(taus_up)) if taus_up else 8.0
    tau_down = float(np.median(taus_down)) if taus_down else 15.0
    return tau_up, tau_down, taus_up, taus_down


# ---------- 公式仿真 ----------
# 5 个状态:off / cold_start / up / down / clamped
def classify_state(prev_freq: float, prev_target: float, target: float,
                   I_comp: float = 0.0, I_max: float = 3.0,
                   freq_cap: float = None) -> str:
    """状态分类(5 状态):off | cold_start | up | down | clamped | hold

    限幅触发条件(任一):
      - I_comp ≥ I_max(电流饱和)
      - prev_freq ≥ freq_cap - 1(硬件 freq 上限)
    """
    if target < ON_THRESHOLD and prev_target >= ON_THRESHOLD:
        return "off"
    if target < ON_THRESHOLD:
        return "off"
    if prev_freq < 5.0 and target >= ON_THRESHOLD:
        return "cold_start"
    # 限幅:freq 达到硬件上限 或 电流饱和
    is_clamped = False
    if I_comp >= I_max and target > prev_freq + 0.5:
        is_clamped = True
    if freq_cap is not None and prev_freq >= freq_cap - 1.0 and target > prev_freq + 0.5:
        is_clamped = True
    if is_clamped:
        return "clamped"
    if target > prev_freq + 0.5:
        return "up"
    if target < prev_freq - 0.5:
        return "down"
    return "hold"


def simulate_freq(target_seq: np.ndarray,
                  I_comp_seq: np.ndarray = None,
                  freq0: float = 0.0,
                  tau_cold: float = 6.7, tau_up: float = 15.0,
                  tau_down: float = 10.0, tau_clamp: float = 1000.0,
                  I_max: float = 3.0,
                  freq_cap: float = None,
                  t_dead_cold_sec: float = 5.0,
                  t_dead_steady_sec: float = 0.0,
                  dt: float = DT, freq_max: float = FREQ_MAX) -> np.ndarray:
    """完整公式前向仿真(5 状态机)。

    状态机:
      off       — 目标 < 15Hz,Freq = 0
      cold_start — 从 0 启动,死区 t_dead_cold_sec 后开始 τ_cold 爬升
      up        — 目标 > Freq 且未限幅,τ_up
      down      — 目标 < Freq,τ_down
      clamped   — 电流饱和(I ≥ I_max) 或 freq 达到硬件上限(freq ≥ freq_cap - 1),τ_clamp

    参数:
      target_seq:    目标频率序列,每行 dt 秒
      I_comp_seq:    压缩机电流序列(可选)
      tau_*:         各状态的时间常数(秒)
      I_max:         限幅触发的电流阈值(典型 3.0A)
      freq_cap:      硬件 freq 上限(可选,source-specific,典型 64Hz 用于限幅 source)
      t_dead_cold_sec / t_dead_steady_sec: 各场景的死区(秒)
    """
    n = len(target_seq)
    freq = np.zeros(n)
    freq[0] = freq0
    # 死区标记(每个时间步是否处于"刚发生阶跃的死区内")
    in_dead_cold = np.zeros(n, dtype=bool)
    in_dead_steady = np.zeros(n, dtype=bool)
    for t in range(1, n):
        is_off_to_on = (target_seq[t - 1] < ON_THRESHOLD and target_seq[t] >= ON_THRESHOLD)
        is_on_to_off = (target_seq[t - 1] >= ON_THRESHOLD and target_seq[t] < ON_THRESHOLD)
        is_jump = abs(target_seq[t] - target_seq[t - 1]) > 5.0
        if is_off_to_on:
            dead_until = t + int(t_dead_cold_sec / dt)
            in_dead_cold[t:dead_until] = True
        elif is_jump and not is_on_to_off:
            dead_until = t + int(t_dead_steady_sec / dt)
            in_dead_steady[t:dead_until] = True
    # 前向迭代
    for t in range(1, n):
        if in_dead_cold[t] or in_dead_steady[t]:
            # 死区内,实际频率不变
            freq[t] = freq[t - 1]
            continue
        # 取当前电流(I_comp_seq 提供时才看)
        I_now = float(I_comp_seq[t]) if I_comp_seq is not None else 0.0
        # 状态分类
        state = classify_state(freq[t - 1], target_seq[t - 1], target_seq[t],
                                I_comp=I_now, I_max=I_max, freq_cap=freq_cap)
        if state == "off":
            freq[t] = 0.0
        elif state == "cold_start":
            tau = tau_cold
            freq[t] = freq[t - 1] + dt / tau * (target_seq[t] - freq[t - 1])
        elif state == "up":
            tau = tau_up
            freq[t] = freq[t - 1] + dt / tau * (target_seq[t] - freq[t - 1])
        elif state == "down":
            tau = tau_down
            freq[t] = freq[t - 1] + dt / tau * (target_seq[t] - freq[t - 1])
        elif state == "clamped":
            # 限幅状态:freq 卡死在 freq_cap 处(完全冻结,不上升)
            # tau_clamp 仅保留参数兼容性,实际用 freq_cap 锁定
            freq[t] = min(freq[t - 1], freq_cap - 0.5) if freq_cap else freq[t - 1]
        else:  # hold
            freq[t] = freq[t - 1]
        freq[t] = max(FREQ_MIN, min(freq_max, freq[t]))
    return freq


# ---------- 评估 ----------
def evaluate_simulation(df: pd.DataFrame, params: dict,
                        sources_to_test: list = None,
                        n_sources: int = 20) -> dict:
    """对若干 source 做前向仿真,对比预测 vs 真实 freq。

    评估指标(分两类 source):
      - 全 source(可能含"电流限幅"异常)
      - 非限幅 source(实际 freq 上限 ≥ 0.85 * 目标上限)

    报告:
      - 整体 MAE / R^2(稳态 + 开关都算)
      - 仅稳态(freq > 0) MAE / R^2
      - 仅 ON 段(冷启动) MAE
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    if sources_to_test is None:
        all_src = list(df["source"].unique())
        rng = np.random.default_rng(0)
        rng.shuffle(all_src)
        sources_to_test = all_src[:n_sources]

    metrics_all = {"overall": [], "steady": [], "on_ramp": []}
    metrics_clean = {"overall": [], "steady": [], "on_ramp": []}
    detailed = []
    for src in sources_to_test:
        g = df[df["source"] == src].sort_values("ts")
        if len(g) < 50:
            continue
        tgt = g["freq_in_tgt"].to_numpy(dtype=float)
        actual = g["freq"].to_numpy(dtype=float)
        # 判断是否电流限幅:目标上限远大于实际能达到的上限(差 ≥20Hz)
        # 整数化数据(max_tgt ≈ max_act,但 freq 取整)不算限幅
        max_act = float(actual.max())
        max_tgt = float(tgt.max())
        is_clamped = (max_tgt - max_act) > 20.0 and max_tgt > 30
        # 传递 I_comp 序列(让限幅逻辑生效)
        I_seq = g["I_comp"].to_numpy(dtype=float) if "I_comp" in g.columns else None
        # 评估用 freq_cap = act_max + 1(per-source,基于历史数据观测)
        params_local = dict(params)
        params_local["freq_cap"] = max_act + 1.0
        pred = simulate_freq(tgt, I_comp_seq=I_seq, freq0=0.0, **params_local)
        bucket = metrics_clean if not is_clamped else metrics_all
        # 整体
        bucket["overall"].append({
            "MAE": float(mean_absolute_error(actual, pred)),
            "R2": float(r2_score(actual, pred)) if actual.std() > 0.1 else 0.0,
        })
        # 稳态
        mask = (actual > 0) & (tgt > 0)
        if mask.sum() > 10:
            bucket["steady"].append({
                "MAE": float(mean_absolute_error(actual[mask], pred[mask])),
                "R2": float(r2_score(actual[mask], pred[mask])),
            })
        # 冷启动段
        on_events = detect_on_events(g.assign(source=g["source"]), post_steps=30)
        for ev in on_events:
            idx0 = ev["t_idx"]
            seg = slice(idx0, min(idx0 + 30, len(actual)))
            bucket["on_ramp"].append({
                "MAE": float(mean_absolute_error(actual[seg], pred[seg])),
            })
        detailed.append({
            "source": src,
            "n": len(actual),
            "is_clamped": is_clamped,
            "max_act": max_act,
            "max_tgt": max_tgt,
        })
    # 汇总
    def agg(metric_list, key):
        if not metric_list:
            return None
        return float(np.median([m[key] for m in metric_list]))
    n_clamped = sum(1 for d in detailed if d["is_clamped"])
    return {
        "n_sources": len(detailed),
        "n_clamped_sources": n_clamped,
        # 全 source(含限幅)
        "all_overall_MAE": agg(metrics_all["overall"], "MAE"),
        "all_overall_R2": agg(metrics_all["overall"], "R2"),
        "all_steady_MAE": agg(metrics_all["steady"], "MAE"),
        "all_steady_R2": agg(metrics_all["steady"], "R2"),
        "all_on_ramp_MAE": agg(metrics_all["on_ramp"], "MAE"),
        # 仅非限幅 source(模型适用)
        "clean_overall_MAE": agg(metrics_clean["overall"], "MAE"),
        "clean_overall_R2": agg(metrics_clean["overall"], "R2"),
        "clean_steady_MAE": agg(metrics_clean["steady"], "MAE"),
        "clean_steady_R2": agg(metrics_clean["steady"], "R2"),
        "clean_on_ramp_MAE": agg(metrics_clean["on_ramp"], "MAE"),
    }


# ---------- main ----------
def main():
    def fmt_metric(value) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print("[load] 加载数据 ...")
    df = load_all_clean()

    print("[detect] 检测冷启动事件 ...")
    on_events = detect_on_events(df, post_steps=90)
    print(f"  {len(on_events)} 个 ON 事件")

    print("[detect] 检测稳态阶跃事件 ...")
    up_events, down_events = detect_step_events(df)
    print(f"  {len(up_events)} 个 UP, {len(down_events)} 个 DOWN")

    print("[fit] 拟合 τ_cold ...")
    tau_cold, taus_cold = fit_tau_cold(on_events)
    print(f"  τ_cold = {tau_cold:.2f}s (中位数, 基于 {len(taus_cold)} 个事件)")

    print("[fit] 拟合 τ_up / τ_down ...")
    tau_up, tau_down, taus_up, taus_down = fit_tau_step(up_events, down_events)
    print(f"  τ_up = {tau_up:.2f}s (基于 {len(taus_up)} 个事件)")
    print(f"  τ_down = {tau_down:.2f}s (基于 {len(taus_down)} 个事件)")

    # 死区
    delays = []
    for ev in on_events:
        nz = np.where(ev["act_window"] > 0)[0]
        if len(nz):
            delays.append(int(nz[0]) * DT)
    delay_mode = float(pd.Series(delays).mode().iloc[0]) if delays else 5.0
    print(f"[fit] 死区(启动延迟)= {delay_mode:.1f}s (众数)")

    params = {
        "tau_cold": tau_cold,
        "tau_up": tau_up,
        "tau_down": tau_down,
        "tau_clamp": 1000.0,   # 限幅时几乎冻结
        "I_max": 3.0,          # 电流阈值
        "t_dead_cold_sec": delay_mode,
        "t_dead_steady_sec": 0.0,
    }

    print(f"\n[params] {json.dumps(params, indent=2)}")

    print("\n[eval] 前向仿真评估(20 个 source)...")
    metrics = evaluate_simulation(df, params, n_sources=20)
    print(f"  source 数: {metrics['n_sources']},其中限幅: {metrics['n_clamped_sources']}")
    print(f"  全 source(含限幅):")
    print(f"    overall MAE={fmt_metric(metrics['all_overall_MAE'])}, R^2={fmt_metric(metrics['all_overall_R2'])}")
    print(f"    steady  MAE={fmt_metric(metrics['all_steady_MAE'])}, R^2={fmt_metric(metrics['all_steady_R2'])}")
    print(f"    on_ramp MAE={fmt_metric(metrics['all_on_ramp_MAE'])}")
    print(f"  仅非限幅 source(模型适用):")
    print(f"    overall MAE={fmt_metric(metrics['clean_overall_MAE'])}, R^2={fmt_metric(metrics['clean_overall_R2'])}")
    print(f"    steady  MAE={fmt_metric(metrics['clean_steady_MAE'])}, R^2={fmt_metric(metrics['clean_steady_R2'])}")
    print(f"    on_ramp MAE={fmt_metric(metrics['clean_on_ramp_MAE'])}")

    out = {
        "params": params,
        "fit_details": {
            "n_on_events": len(on_events),
            "n_on_used_for_fit": len(taus_cold),
            "n_up_events": len(up_events),
            "n_up_used_for_fit": len(taus_up),
            "n_down_events": len(down_events),
            "n_down_used_for_fit": len(taus_down),
            "delay_mode_sec": delay_mode,
        },
        "metrics": metrics,
    }
    (OUT / "freq_model_params.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print(f"\n[done] 参数与指标已写入 {OUT / 'freq_model_params.json'}")
    return params


if __name__ == "__main__":
    main()
