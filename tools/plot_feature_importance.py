"""生成四灾害特征重要性图"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import numpy as np
from models.inference import DisasterInference
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"outputs","feature_importance.png")
engine=DisasterInference()
fig,axes=plt.subplots(2,2,figsize=(16,12))
nm={'flash_flood':'暴雨山洪','extreme_heat':'极端高温','dust_wind':'沙尘强风','coastal_wave':'沿海风浪'}
cs=['#ef4444','#f97316','#eab308','#3b82f6']
for ax,(dtype,color) in zip(axes.flat,zip(nm.keys(),cs)):
    m=engine.models[dtype]; names=m.model.feature_name(); imp=m.model.feature_importance(importance_type='gain')
    n=min(10,len(imp)); idx=np.argsort(imp)[-n:]; vals=imp[idx]; lbls=[names[i] for i in idx]
    ax.barh(range(n),vals,color=color); ax.set_yticks(range(n)); ax.set_yticklabels(lbls,fontsize=10)
    ax.set_title(nm[dtype],fontsize=14,fontweight='bold')
    for i,v in enumerate(vals):ax.text(v+max(vals)*.02,i,f'{v:.0f}',va='center',fontsize=9)
fig.suptitle('MAZU 四灾害 LightGBM 特征重要性 Top-10 (Gain)',fontsize=18,fontweight='bold',y=.98)
plt.tight_layout();fig.savefig(OUT,dpi=150,bbox_inches='tight')
print(f'✅ {OUT}')
