# -*- coding: utf-8 -*-
"""PINN (Physics-Informed Neural Network) for AC power prediction."""
from pathlib import Path
import torch, torch.nn as nn, numpy as np

FEATURE_COLS = ['压缩机运行频率', '电子膨胀阀开度', '室外风机转速']


class PINN(nn.Module):
    """P = baseline + softplus(freq_net(freq)) + valve_net(valve) + fan_net(fan) + cross_net(all)"""
    def __init__(self):
        super().__init__()
        self.freq_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        self.valve_net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        self.fan_net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        self.cross_net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))
        self.baseline = nn.Parameter(torch.tensor([20.0]))

    def forward(self, x):
        f, v, r = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        f_term = nn.functional.softplus(self.freq_net(f))
        v_term = self.valve_net(v)
        r_term = self.fan_net(r)
        cross = self.cross_net(x)
        return (self.baseline + f_term + v_term + r_term + cross).squeeze(-1)


def load_pinn(path=None):
    """Load PINN model + scalers. Returns dict with model, sx, sy, baseline."""
    if path is None:
        path = Path(__file__).resolve().with_name('torch_pinn.pth')
    ckpt = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
    model = PINN()
    model.load_state_dict(ckpt['model'])
    model.eval()
    return {
        'model': model,
        'sx': ckpt['sx'],
        'sy': ckpt['sy'],
        'baseline': float(ckpt['baseline']),
    }


def predict_pinn(pinn_ckpt, freq, valve, fan):
    """Predict power (W) from (freq, valve, fan)."""
    x = np.array([[freq, valve, fan]], dtype=np.float32)
    x_s = pinn_ckpt['sx'].transform(x)
    with torch.no_grad():
        y_s = pinn_ckpt['model'](torch.tensor(x_s)).numpy()
    return float(pinn_ckpt['sy'].inverse_transform(y_s.reshape(-1, 1))[0, 0])


if __name__ == '__main__':
    ckpt = load_pinn()
    print(f"PINN loaded. baseline = {ckpt['baseline']:.1f} W")
    print(f"predict(30, 250, 520) = {predict_pinn(ckpt, 30, 250, 520):.1f} W")  # training range
    print(f"predict(150, 300, 750) = {predict_pinn(ckpt, 150, 300, 750):.1f} W")  # extrapolation test
