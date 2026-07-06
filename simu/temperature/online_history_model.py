"""5秒在线历史统计模型：仅使用截至当前的控制前缀。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def prefix_feature(state0: np.ndarray, prefix: np.ndarray, step: int, full_horizon: int = 720, blocks: int = 6) -> np.ndarray:
    chunks = np.array_split(prefix, min(blocks, len(prefix)))
    means = np.stack([c.mean(0) for c in chunks])
    if len(means) < blocks:
        means = np.concatenate([means, np.repeat(means[-1:], blocks-len(means), axis=0)])
    return np.concatenate([
        state0,
        [step/full_horizon],
        means.reshape(-1),
        prefix.mean(0), prefix.std(0), prefix.min(0), prefix.max(0), prefix[0], prefix[-1],
    ]).astype(np.float32)


def training_matrix(states: np.ndarray, controls: np.ndarray, horizons: tuple[int,...], blocks: int = 6):
    x=[];y=[]
    for s,u in zip(states,controls):
        for h in horizons:
            x.append(prefix_feature(s[0],u[:h],h,controls.shape[1],blocks));y.append(s[h,2]-s[0,2])
    return np.stack(x),np.asarray(y,np.float32)


@dataclass
class OnlineHistoryEnsemble:
    members: list
    blocks: int = 6
    full_horizon: int = 720
    smoothing_tau_seconds: float = 30.0

    def predict_trajectory(self, states: np.ndarray, controls: np.ndarray):
        outputs=[];spreads=[]
        for s,u in zip(states,controls):
            feats=np.stack([prefix_feature(s[0],u[:h],h,self.full_horizon,self.blocks) for h in range(1,len(u)+1)])
            member=np.stack([m.predict(feats) for m in self.members])
            raw=np.r_[s[0,2],s[0,2]+member.mean(0)]
            smooth=raw.copy();alpha=5.0/(self.smoothing_tau_seconds+5.0)
            for t in range(1,len(smooth)):smooth[t]=smooth[t-1]+alpha*(raw[t]-smooth[t-1])
            outputs.append(smooth)
            spreads.append(np.r_[0,member.std(0)])
        return np.stack(outputs),np.stack(spreads)
