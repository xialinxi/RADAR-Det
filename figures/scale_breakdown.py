import matplotlib.pyplot as plt

# 数据定义
methods = ['PGD-AT', 'Free-AT', 'NAG-LR', 'RADAR-Det (Ours)']
times = [45, 18, 24, 21]  # 训练时间 (h)
maps = [23.8, 12.5, 29.3, 37.1] # 鲁棒 mAP (%)
colors = ['#34495e', '#95a5a6', '#2980b9', '#e74c3c']
sizes = [100, 100, 150, 300] # 突出展示我们的方法

plt.figure(figsize=(8, 6), dpi=300)
for i in range(len(methods)):
    plt.scatter(times[i], maps[i], s=sizes[i], color=colors[i], label=methods[i], edgecolors='black', alpha=0.8)
    plt.text(times[i]+0.8, maps[i]+0.5, methods[i], fontsize=10, fontweight='bold' if 'Ours' in methods[i] else 'normal')

plt.xlabel('Total Training Time (Hours)', fontsize=12)
plt.ylabel('Robust mAP @[0.5:0.95] (%)', fontsize=12)
plt.title('Efficiency-Robustness Pareto Frontier', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(15, 50)
plt.ylim(10, 42)
plt.axvspan(15, 22, color='green', alpha=0.05) # 高效区
plt.text(16, 38, "High Efficiency Zone", color='green', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig('efficiency_tradeoff.png')
plt.show()