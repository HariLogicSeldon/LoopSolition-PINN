import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
y = np.linspace(-1.5, 1.5, 180)
t = np.linspace(-1.5, 1.5, 180)
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
# Save to the current directory or specify an absolute path
output_path = 'data/LoopSolition.mat'  # Change path as needed
scipy.io.savemat(output_path, output_data)

data = scipy.io.loadmat(output_path)
y = data['y'].flatten()
t = data['t'].flatten()
xx_data = data['xx']
qq_data = data['qq']

# 创建网格
T, Y = np.meshgrid(t, y)

# Step 5: Plot the results
XX = xx(Y, T).real  # Re(x)
QQ = np.abs(qq(Y, T))  # |q|

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 Re(x), t, |q| 的 surface
surface = ax.plot_surface(T, XX, QQ, cmap='plasma', edgecolor='none', alpha=0.8, label="Re(x)")

# 添加等高热力图
contour = ax.contourf(T, XX, QQ, zdir='z', offset=np.min(QQ) - 0.1, cmap='plasma', alpha=0.8)  # 设置热力图投影到 z 平面

# 添加颜色条
cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('|q|', fontsize=12)

# 设置轴标签
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('Re(X)', fontsize=12)
ax.set_zlabel('|q|', fontsize=12)
ax.set_title('3D Plot of Re(x) and |q| with t and Heatmap Projection', fontsize=14)

# 设置 z 轴范围（可选，根据数据手动调整）
ax.set_zlim(np.min(QQ) - 0.1, np.max(QQ) )


# Step 6: Generate 2D plots for fixed t and y showing the relationship between Re(x) and |q|

fixed_t = [-10, 0, 10]  # Fixed t values
fixed_y = np.linspace(-1.5, 1.5, 100)  # Range of y values
line_styles = ['--', '-.', ':']  # Different line styles for each t

# Create a figure for 2D plots
plt.figure(figsize=(10, 6))

# Compute and plot Re(x) vs |q| for each fixed t with different line styles
for t_point, style in zip(fixed_t, line_styles):
    # Compute Re(x) and |q| at the fixed t and range of y
    x_fixed_t = xx(fixed_y, t_point).real  # Re(x) at fixed t
    q_fixed_t = np.abs(qq(fixed_y, t_point))  # |q| at fixed t
    print(x_fixed_t.shape)
    print(q_fixed_t.shape)

    plt.plot(x_fixed_t, q_fixed_t, label=f"t={t_point}", linestyle=style)

# Add labels, legend, and title
plt.xlabel('Re(x)')
plt.ylabel('|q|')
plt.title('2D Plot of Re(x) vs |q| at Fixed t and Varying y with Different Line Styles')
plt.legend()
plt.grid()

# Display the plot
plt.show()

# Print confirmation of saved file
print(f'Data saved to {output_path}')

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
