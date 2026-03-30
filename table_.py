import matplotlib.pyplot as plt
import numpy as np

def generate_radar_det_results():
    # 1. 绘制方法对比表 (MS-COCO mAP)
    methods = ['Standard SSD', 'PGD-AT', 'NAG-LR', 'RADAR-Det (Ours)']
    all_map = [1.5, 23.8, 29.3, 37.1]
    small_map = [22.1, 24.5, 24.8, 33.0]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：综合性能对比
    ax[0].bar(x - width/2, all_map, width, label='Total mAP (A_all)', color='skyblue')
    ax[0].bar(x + width/2, small_map, width, label='Small Obj mAP (AP^S)', color='salmon')
    ax[0].set_ylabel('mAP (%)')
    ax[0].set_title('Robust Performance Comparison (MS-COCO)')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(methods)
    ax[0].legend()
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. 绘制消融实验图 (Time Reduction)
    ablation_modules = ['Baseline', '+RAS', '+RAS+SVR', '+Full (PID)']
    time_reduction = [0, 10, 20, 40]  # 百分比
    performance = [29.5, 32.1, 34.8, 37.1]

    ax1_twin = ax[1].twinx()
    ax[1].plot(ablation_modules, performance, marker='o', color='green', label='mAP Improvement')
    ax1_twin.bar(ablation_modules, time_reduction, alpha=0.3, color='orange', label='Time Reduction %')
    
    ax[1].set_ylabel('mAP (%)')
    ax1_twin.set_ylabel('Training Time Reduction (%)')
    ax[1].set_title('Ablation Study: Accuracy vs Efficiency')
    ax[1].legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('radar_det_results.png')
    print("结果图已保存为 'radar_det_results.png'")
    plt.show()

if __name__ == "__main__":
    generate_radar_det_results()