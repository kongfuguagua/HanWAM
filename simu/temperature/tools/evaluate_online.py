"""最终5秒在线环境的严格留出测试、逐步一致性和耗时评估。"""
from pathlib import Path
import json,time,joblib
PROJECT=Path(__file__).resolve().parent.parent
ROOT=PROJECT.parents[1]
RESULTS=ROOT/'data'/'results'/'temperature'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simu.temperature.tools.data_pipeline import load_all_runs,STATE_COLS,CONTROL_COLS,describe_runs
from simu.temperature.online_history_model import OnlineHistoryEnsemble
from simu.temperature.simulator import OnlineEnthalpyRoomEnv
from simu.temperature.tools.online_utils import start_windows,temperature_metrics

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    runs=load_all_runs(5);describe_runs(runs,5).to_csv(RESULTS/'online_data_audit.csv',index=False,encoding='utf-8-sig');s,u,names=start_windows(runs,'test');payload=joblib.load(PROJECT/'online_history_ensemble.joblib')
    pred,spread=OnlineHistoryEnsemble(payload['members'],payload['blocks']).predict_trajectory(s,u);m=temperature_metrics(pred,s)
    # 对第一段逐行调用，确认批量研究实现与生产step接口完全一致。
    env=OnlineEnthalpyRoomEnv();online=env.simulate([s[0,0,0],s[0,0,2],s[0,0,1],s[0,0,3]],u[0]);max_diff=float(np.max(np.abs(online.T_in.to_numpy()-pred[0])))
    env.reset(s[0,0,0],s[0,0,2],s[0,0,1],s[0,0,3]);t0=time.perf_counter()
    for row in u[0]:env.step(*row)
    ms_per_step=(time.perf_counter()-t0)*1000/len(u[0])
    report={'protocol':{'test_runs':8,'step_seconds':5,'steps':720,'split':'第3轮完整实验，未参与训练或选参','causal':True},'metrics':m,'online_check':{'batch_step_max_difference_c':max_diff,'mean_inference_ms_per_step':ms_per_step}}
    (RESULTS/'online_evaluation_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[];long=[]
    for i,name in enumerate(names):
        err=pred[i]-s[i,:,2];rows.append({'run':name,'trajectory_mae_c':abs(err[1:]).mean(),'actual_60min_c':s[i,-1,2],'pred_60min_c':pred[i,-1],'abs_error_60min_c':abs(err[-1])})
        for t in range(721):long.append({'run':name,'elapsed_seconds':t*5,'actual_T_in':s[i,t,2],'pred_T_in':pred[i,t],'error_c':err[t]})
    pd.DataFrame(rows).to_csv(RESULTS/'online_per_run_metrics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(long).to_csv(RESULTS/'online_test_trajectories.csv',index=False,encoding='utf-8-sig')
    x=np.arange(721)*5/60;fig,axs=plt.subplots(4,2,figsize=(13,12),sharex=True)
    for ax,i in zip(axs.flat,range(8)):
        ax.plot(x,s[i,:,2],label='Actual',lw=1.8);ax.plot(x,pred[i],label='Online 5s',lw=1.5);ax.set_title(names[i].split('_status')[0]);ax.grid(alpha=.25);ax.set_ylabel('T_in (°C)')
    axs.flat[0].legend();axs[-1,0].set_xlabel('Minutes');axs[-1,1].set_xlabel('Minutes');fig.suptitle('Held-out round 3: true 5-second online simulation');fig.tight_layout();fig.savefig(RESULTS/'online_heldout_first_hour.png',dpi=180);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
