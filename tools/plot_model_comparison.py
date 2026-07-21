"""生成四灾害模型性能对比图"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import numpy as np
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False

OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"outputs","model_comparison.png")

# 四灾害模型性能数据
disasters = ['暴雨山洪', '极端高温', '沙尘强风', '沿海风浪']
colors = ['#ef4444', '#f97316', '#eab308', '#3b82f6']
features = [25, 24, 19, 15]
csi = [1.0, 1.0, 1.0, 1.0]
pod = [1.0, 1.0, 1.0, 1.0]
far = [0.0, 0.0, 0.0, 0.0]
thresholds = [0.50, 0.95, 0.50, 0.95]
auc = [1.0, 1.0, 1.0, 1.0]
# 真实事件验证
real_hits = ['5/5 (100%)', '2/2 (100%)', '2/3 (67%)', '—']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：柱状图 - 特征数 + 阈值
x = np.arange(len(disasters))
w = 0.35
bars1 = ax1.bar(x - w/2, features, w, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + w/2, thresholds, w, color=['#64748b']*4, alpha=0.5, edgecolor='white', linewidth=1.5)
ax1.set_ylabel('特征数量', fontsize=12, color='#334155')
ax1_twin.set_ylabel('判定阈值', fontsize=12, color='#64748b')
ax1.set_xticks(x); ax1.set_xticklabels(disasters, fontsize=11)
ax1.set_title('特征维度 & 判定阈值', fontsize=15, fontweight='bold', color='#1e293b')
for bar, v in zip(bars1, features): ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, str(v), ha='center', fontsize=11, fontweight='bold', color='#334155')
for bar, v in zip(bars2, thresholds): ax1_twin.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, str(v), ha='center', fontsize=10, color='#64748b')

# 右图：CSI/POD/FAR 热力表格
cell_text = [
    [f'{csi[i]:.3f}', f'{pod[i]:.3f}', f'{far[i]:.3f}', f'{auc[i]:.3f}', f'{thresholds[i]}', f'{features[i]}', real_hits[i]]
    for i in range(4)
]
columns = ['CSI', 'POD', 'FAR', 'AUC', '阈值', '特征数', '真实事件命中']
ax2.axis('tight'); ax2.axis('off')
table = ax2.table(cellText=cell_text, colLabels=columns, rowLabels=disasters,
    rowColours=colors, colColours=['#f1f5f9']*7, cellLoc='center', loc='center')
table.auto_set_font_size(False); table.set_fontsize(11)
table.scale(1.2, 2.0)
for i in range(4):
    for j in range(7):
        table[i+1, j].set_facecolor('#f8fafc' if j < 6 else '#fef3c7')
for j in range(7): table[0, j].set_facecolor('#e2e8f0'); table[0, j].set_fontsize(10)
for i in range(4): table[i+1, 6].set_facecolor('#fef3c7')
ax2.set_title('四灾害 LightGBM 模型性能汇总', fontsize=15, fontweight='bold', color='#1e293b')

fig.suptitle('MAZU 沙特多灾种预警 — 模型性能总览', fontsize=18, fontweight='bold', y=1.01, color='#0f172a')
plt.tight_layout(); fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f'✅ {OUT}')
