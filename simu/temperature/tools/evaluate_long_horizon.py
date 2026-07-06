"""保持模型不变，测试从实验起点连续在线外推1/2/3小时的漂移。"""
from pathlib import Path
import json,joblib
PROJECT=Path(__file__).resolve().parent.parent
ROOT=PROJECT.parents[1]
RESULTS=ROOT/'data'/'results'/'temperature'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simu.temperature.tools.data_pipeline import load_all_runs
from simu.temperature.online_history_model import OnlineHistoryEnsemble
from simu.temperature.tools.online_utils import start_windows


def horizon_metrics(pred,truth,step):
    err=pred[:,:step+1]-truth[:,:step+1,2];final=np.abs(err[:,-1])
    return {'hours':step*5/3600,'trajectory_mae_c':float(np.abs(err[:,1:]).mean()),'final_mae_c':float(final.mean()),'final_median_ae_c':float(np.median(final)),'final_p90_ae_c':float(np.quantile(final,.9)),'within_0_5c_rate':float((final<=.5).mean()),'mean_signed_drift_c':float(err[:,-1].mean()),'rmse_c':float(np.sqrt(np.mean(err[:,1:]**2)))}


def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    runs=load_all_runs(5);states,controls,names=start_windows(runs,'test',horizon=2160)
    payload=joblib.load(PROJECT/'online_history_ensemble.joblib')
    model=OnlineHistoryEnsemble(payload['members'],payload['blocks'],full_horizon=720)
    pred,_=model.predict_trajectory(states,controls)
    steps=[720,1440,2160];summary={f'{h//720}h':horizon_metrics(pred,states,h) for h in steps}
    (RESULTS/'long_horizon_metrics.json').write_text(json.dumps({'note':'模型未重新训练；2/3小时均为超出1小时训练范围的外推。','metrics':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for i,name in enumerate(names):
        row={'run':name}
        for h in steps:
            label=f'{h//720}h';e=float(pred[i,h]-states[i,h,2]);row[f'actual_{label}_c']=float(states[i,h,2]);row[f'pred_{label}_c']=float(pred[i,h]);row[f'signed_drift_{label}_c']=e;row[f'abs_drift_{label}_c']=abs(e)
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS/'long_horizon_per_run.csv',index=False,encoding='utf-8-sig')
    long=[]
    for i,name in enumerate(names):
        for t in range(2161):long.append({'run':name,'elapsed_seconds':t*5,'actual_T_in':states[i,t,2],'pred_T_in':pred[i,t],'error_c':pred[i,t]-states[i,t,2]})
    pd.DataFrame(long).to_csv(RESULTS/'long_horizon_trajectories.csv',index=False,encoding='utf-8-sig')
    x=np.arange(2161)*5/3600;fig,axs=plt.subplots(4,2,figsize=(13,12),sharex=True)
    for ax,i in zip(axs.flat,range(8)):
        ax.plot(x,states[i,:,2],label='Actual',lw=1.7);ax.plot(x,pred[i],label='Online model',lw=1.4)
        ax.axvline(1,color='#999999',ls='--',lw=.8);ax.axvline(2,color='#999999',ls='--',lw=.8);ax.set_title(names[i].split('_status')[0]);ax.grid(alpha=.22);ax.set_ylabel('T_in (°C)')
    axs.flat[0].legend();axs[-1,0].set_xlabel('Hours');axs[-1,1].set_xlabel('Hours');fig.suptitle('Held-out round 3: unchanged model extrapolated to 3 hours');fig.tight_layout();fig.savefig(RESULTS/'long_horizon_1h_2h_3h.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
