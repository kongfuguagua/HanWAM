"""5秒在线模型共用的数据窗口和评估指标。"""
import numpy as np
from simu.temperature.tools.data_pipeline import STATE_COLS,CONTROL_COLS


def start_windows(runs,split,horizon=720):
    selected=[r for r in runs if r.split==split]
    states=np.stack([r.frame[STATE_COLS].to_numpy(np.float32)[:horizon+1] for r in selected])
    controls=np.stack([r.frame[CONTROL_COLS].to_numpy(np.float32)[:horizon] for r in selected])
    return states,controls,[r.name for r in selected]


def temperature_metrics(pred,truth):
    err=pred-truth[:,:,2];final=np.abs(err[:,-1])
    return {'trajectory_mae_c':float(np.abs(err[:,1:]).mean()),'final_60min_mae_c':float(final.mean()),'final_60min_median_ae_c':float(np.median(final)),'final_60min_p90_ae_c':float(np.quantile(final,.9)),'within_0_5c_rate':float((final<=.5).mean()),'rmse_c':float(np.sqrt(np.mean(err[:,1:]**2)))}
