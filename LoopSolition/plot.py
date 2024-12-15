import matplotlib.pyplot as plt
import numpy as np
from pinnstorch.utils.plotting import *

def plot_loop_solition(mesh, preds,file_name):
    """Plot Loop Solition plot"""
    """
        绘制 Re(xx)、|qq| 和 y 在不同时间点的行为。

        :param mesh: 网格对象，包含真实解。
        :param preds_dict: 预测结果字典，来自模型预测。
        :param time_points: 要绘制的时间点（列表或数组）。
        :param file_name: 保存图片的文件名。
        """

    y = mesh.spatial_domain[0].flatten()  # y.shape = (180,)
    t = mesh.time_domain.flatten()  # t.shape = (180,)

    # 从预测数据中已得出：
    xx_pred = preds["xx"].view(180, 180).numpy()
    qq_pred = preds["qq"].view(180, 180).numpy()
    xx_pred_real = xx_pred.real
    qq_pred_modulus = np.abs(qq_pred)
    print("xx_pred_real", xx_pred_real.shape)
    print("qq_pred_modulus", qq_pred_modulus.shape)

    # 从真实数据中已得出：
    xx_exact = np.array(mesh.solution["xx"])  # (180,180)
    qq_exact = np.array(mesh.solution["qq"])  # (180,180)

    xx_real = xx_exact.real
    qq_modulus = np.abs(qq_exact)
    print("xx_real", xx_real.shape)
    print("qq_modulus", qq_modulus.shape)

    # 固定的 t 值（与选中的代码段相似）
    fixed_t = [-1.5, 0, 1.5]
    line_styles = ['--', '-.', ':']

    plt.figure(figsize=(10, 6))

    for t_point, style in zip(fixed_t, line_styles):
        # 根据固定的 t 点，在 t 数组中找到最接近的索引
        t_idx = np.argmin(np.abs(t - t_point))

        # 提取真实数据下的 Re(x) 与 |q|
        x_fixed_t_exact = xx_real[:,t_idx] # 固定t下真实的Re(x)在y方向上的分布
        q_fixed_t_exact = qq_modulus[:,t_idx]  # 固定t下真实的|q|在y方向上的分布
        # 绘制真实数据的曲线
        plt.plot(x_fixed_t_exact, q_fixed_t_exact, label=f"Real t={t_point}", linestyle=style)

        # # 同样提取预测数据下的 Re(x) 与 |q|
        x_fixed_t_pred = xx_pred_real[:,t_idx]
        q_fixed_t_pred = qq_pred_modulus[:,t_idx]

        # 绘制预测数据的曲线 (可以用不同颜色或标记区分)
        plt.plot(x_fixed_t_pred, q_fixed_t_pred, label=f"Pred t={t_point}", linestyle=style, marker='*', markevery=15)

    plt.xlabel('Re(x)')
    plt.ylabel('|q|')
    # plt.title('在固定 t 下，真实值与预测值的 Re(x) vs |q| 比较')
    plt.legend()
    plt.grid(True)

# #### 绘制模的热图
    plt.figure(figsize=(12, 6))

    # xx 模的原始值
    plt.subplot(2, 2, 1)
    plt.imshow(xx_real, cmap='viridis', extent=[t.min(), t.max(), y.min(), y.max()])
    plt.colorbar()
    plt.title('Re(x) Exact')

    # xx 模的预测值
    plt.subplot(2, 2, 2)
    plt.imshow(xx_pred_real, cmap='viridis', extent=[t.min(), t.max(), y.min(), y.max()])
    plt.colorbar()
    plt.title('Re(x) Predicted')

    # qq 模的原始值
    plt.subplot(2, 2, 3)
    plt.imshow(qq_modulus, cmap='viridis', extent=[t.min(), t.max(), y.min(), y.max()])
    plt.colorbar()
    plt.title('|qq| Exact')

    # qq 模的预测值
    plt.subplot(2, 2, 4)
    plt.imshow(qq_pred_modulus, cmap='viridis', extent=[t.min(), t.max(), y.min(), y.max()])
    plt.colorbar()
    plt.title('|qq| Predicted')

    plt.tight_layout()
    plt.show()
