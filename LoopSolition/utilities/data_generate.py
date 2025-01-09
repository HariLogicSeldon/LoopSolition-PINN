import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os  # 新增导入 os 模块

# Step 1: Define the functions for xx(y, t) and qq(y, t)
def xx(y, t):
    numerator = ((2 + 1j) + 5 * y) * np.exp(-4 * y - t / 5) + ((-2 + 1j) + 5 * y) * np.exp(4 * y + t / 5)
    denominator = 5 * np.exp(4 * y + t / 5) + 5 * np.exp(-4 * y - t / 5)
    return numerator / denominator

def qq(y, t):
    numerator = 4 * np.exp(-1j / 10 * (t - 20 * y))
    denominator = 5 * np.exp(4 * y + t / 5) + 5 * np.exp(-4 * y - t / 5)
    return numerator / denominator

# Step 2: Generate y and t values
y = np.linspace(-1.5, 1.5, 256)
t = np.linspace(-0.5, 0.5, 128)
Y, T = np.meshgrid(y, t)

# Step 3: Compute the xx and qq values
XX = xx(Y, T)
QQ = qq(Y, T)

# Step 4: Save the data to a .mat file
output_data = {
    'y': y,
    't': t,
    'xx': XX,
    'qq': QQ
}
#获取当前脚本所在目录的父目录（即项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
data_dir = os.path.join(root_dir, 'data')

# 如果 data 目录不存在，则创建它
os.makedirs(data_dir, exist_ok=True)


# 设置保存路径到根目录的 data 文件夹
output_path = os.path.join(data_dir, 'LoopSolition.mat')
scipy.io.savemat(output_path, output_data)




# Step 5: 从 mat 文件加载数据进行绘图
# 加载保存的数据
data = scipy.io.loadmat(output_path)
y = data['y'].flatten()
t = data['t'].flatten()
xx_data = data['xx'].T
qq_data = data['qq'].T

# 创建网格
T, Y = np.meshgrid(t, y)

# 准备绘图数据
XX_plot = xx_data.real  # Re(x)
QQ_plot = np.abs(qq_data)  # |q|

# 创建3D图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 Re(x), t, |q| 的 surface
surface = ax.plot_surface(T, XX_plot, QQ_plot, cmap='plasma',
                         edgecolor='none', alpha=0.8)

# 添加等高热力图
contour = ax.contourf(T, XX_plot, QQ_plot, zdir='z',
                      offset=np.min(QQ_plot) - 0.1,
                      cmap='plasma', alpha=0.8)

# 添加颜色条
cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('|q|', fontsize=12)

# 设置轴标签
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('Re(X)', fontsize=12)
ax.set_zlabel('|q|', fontsize=12)
ax.set_title('3D Plot of Re(x) and |q| with t and Heatmap Projection',
             fontsize=14)

# 设置 z 轴范围
ax.set_zlim(np.min(QQ_plot) - 0.1, np.max(QQ_plot))

# Step 6: 生成固定 t 值的 2D 图
fixed_t_indices = [0, len(t)//2, -1]  # 选择开始、中间和结束的时间点
line_styles = ['--', '-.', ':']

plt.figure(figsize=(10, 6))

# 对每个固定的 t 值绘制图形
for t_idx, style in zip(fixed_t_indices, line_styles):
    t_value = t[t_idx]
    x_fixed_t = xx_data[:, t_idx].real  # 在固定 t 处的 Re(x)
    q_fixed_t = np.abs(qq_data[:, t_idx])  # 在固定 t 处的 |q|

    plt.plot(x_fixed_t, q_fixed_t,
             label=f"t={t_value:.2f}",
             linestyle=style)

# 添加标签和图例
plt.xlabel('Re(x)')
plt.ylabel('|q|')
plt.title('2D Plot of Re(x) vs |q| at Fixed t and Varying y')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
