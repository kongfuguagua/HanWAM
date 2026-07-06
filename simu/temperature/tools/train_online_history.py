"""训练低漂移的5秒在线历史统计集成。"""
from pathlib import Path
import json,joblib
PROJECT=Path(__file__).resolve().parent.parent
ROOT=PROJECT.parents[1]
RESULTS=ROOT/'data'/'results'/'temperature'
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor

from simu.temperature.tools.data_pipeline import *
from simu.temperature.online_history_model import OnlineHistoryEnsemble,training_matrix
from simu.temperature.tools.online_utils import start_windows,temperature_metrics

H=tuple(range(6,721,6))  # 每30秒一个训练监督点，推理仍可每5秒计算。

def build(kind,leaf,seed):
    if kind=='hist':return HistGradientBoostingRegressor(max_iter=300,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=leaf,l2_regularization=.1,random_state=seed)
    cls=ExtraTreesRegressor if kind=='extra' else RandomForestRegressor
    return cls(n_estimators=120,min_samples_leaf=leaf,max_features=1.0,n_jobs=-1,random_state=seed)

def main():
    runs=load_all_runs(5);ts,tu,_=make_windows(runs,'train',720,60);vs,vu,_=start_windows(runs,'val')
    rows=[];models={}
    for blocks in (4,6,8):
        x,y=training_matrix(ts,tu,H,blocks)
        for kind,leaves in [('hist',(50,)),('extra',(4,)),('rf',(4,))]:
            for leaf in leaves:
                model=build(kind,leaf,2026).fit(x,y);pred,_=OnlineHistoryEnsemble([model],blocks).predict_trajectory(vs,vu);m=temperature_metrics(pred,vs)
                row={'blocks':blocks,'kind':kind,'leaf':leaf,**m};rows.append(row);models[(blocks,kind,leaf)]=model;print(json.dumps(row,ensure_ascii=False),flush=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(rows).sort_values(['final_60min_mae_c','trajectory_mae_c']);table.to_csv(RESULTS/'online_history_tuning.csv',index=False)
    print('best validation\n',table.head(5).to_string(index=False),flush=True)
    # 选定结构后合并验证轮次重新训练；测试轮次保持完全留出。
    vals,valu,_=make_windows(runs,'val',720,60);ls=np.concatenate([ts,vals]);lu=np.concatenate([tu,valu])
    configs=table.head(3).to_dict('records');members=[]
    # 为保证step接口统一，最终三个成员采用相同blocks；选择最佳blocks下的前三名。
    best_blocks=int(table.iloc[0].blocks);configs=table[table.blocks==best_blocks].head(1).to_dict('records')
    x,y=training_matrix(ls,lu,H,best_blocks)
    for seed,c in enumerate(configs,3001):members.append(build(c['kind'],int(c['leaf']),seed).fit(x,y))
    payload={'members':members,'blocks':best_blocks,'configs':configs};joblib.dump(payload,PROJECT/'online_history_ensemble.joblib',compress=3)
    ss,su,names=start_windows(runs,'test');pred,spread=OnlineHistoryEnsemble(members,best_blocks).predict_trajectory(ss,su);result=temperature_metrics(pred,ss);print('test',json.dumps(result,ensure_ascii=False),flush=True)
    pd.DataFrame({'run':names,'actual_60min_c':ss[:,-1,2],'pred_60min_c':pred[:,-1],'abs_error_c':abs(pred[:,-1]-ss[:,-1,2]),'spread_c':spread[:,-1]}).to_csv(RESULTS/'online_history_test_start.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':main()
