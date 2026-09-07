import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set(style='whitegrid')
colors = sns.color_palette('Set2', n_colors=10)

# 配置
base_dir = 'RogueWave/lightning_logs/csv_logs'
output_dir = 'RogueWave/figures'
os.makedirs(output_dir, exist_ok=True)
metrics = ['val/error_p', 'val/error_q', 'val/error_u', 'val/error_v']
smoothing_window = 20  # 平滑窗口大小

# 获取所有 version_x 文件夹
version_dirs = [d for d in os.listdir(base_dir) if d.startswith('version_')]
version_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # 按版本号排序

# 准备画图
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    found_data = False

    for j, version in enumerate(version_dirs):
        csv_path = os.path.join(base_dir, version, 'metrics.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        if metric not in df.columns or 'step' not in df.columns:
            continue

        # 数据清洗 + 平滑
        cleaned = df[['step', metric]].dropna()
        # cleaned = cleaned[cleaned[metric] < 1e5]
        cleaned = cleaned.sort_values('step').reset_index(drop=True)

        if not cleaned.empty:
            # 平滑曲线
            smoothed = cleaned[metric].rolling(window=smoothing_window, min_periods=1).mean()
            ax.plot(
                cleaned['step'], smoothed,
                label=version,
                color=colors[j % len(colors)],
                linewidth=2,
                alpha=0.9
            )
            found_data = True

    ax.set_title(metric, fontsize=14)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    if found_data:
        ax.legend(fontsize=9, loc='upper right', frameon=False)
    else:
        ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()

# 保存图像
# png_path = os.path.join(output_dir, 'error_metrics.png')
pdf_path = os.path.join(output_dir, 'error_metrics.pdf')
# plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)

plt.show()
print(f"✅ 图像已保存为：\n- {pdf_path}")