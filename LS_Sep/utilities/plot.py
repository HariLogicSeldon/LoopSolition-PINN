import matplotlib.pyplot as plt
import numpy as np
from pinnstorch.utils.plotting import *

def plot_loop_solition(mesh, preds, file_name):
    """Plot Loop Solition plot"""
    """
        绘制 Re(xx)、|qq| 和 y 在不同时间点的行为。

        :param mesh: 网格对象，包含真实解。
        :param preds: 预测结果字典，包含 u, v, p, q
        :param file_name: 保存图片的文件名。
    """

    y = mesh.spatial_domain[0].flatten()  # y.shape = (180,)
    t = mesh.time_domain.flatten()  # t.shape = (180,)

    # 从预测数据中提取实部和虚部
    u_pred = preds["u"].view(256, 256).numpy()
    v_pred = preds["v"].view(256, 256).numpy()
    p_pred = preds["p"].view(256, 256).numpy()
    q_pred = preds["q"].view(256, 256).numpy()

    # 计算 qq 的模长
    qq_pred_modulus = np.sqrt(p_pred**2 + q_pred**2)

    # 从真实数据中提取实部和虚部
    u_exact = np.array(mesh.solution["u"])  # (180,180)
    v_exact = np.array(mesh.solution["v"])  # (180,180)
    p_exact = np.array(mesh.solution["p"])  # (180,180)
    q_exact = np.array(mesh.solution["q"])  # (180,180)

    # 计算 qq 的模长
    qq_exact_modulus = np.sqrt(p_exact**2 + q_exact**2)

    # 固定的 t 值（根据具体情况调整）
    fixed_t = [-1.5, 0, 1.5]
    line_styles = ['--', '-.', ':']

    plt.figure(figsize=(10, 6))

    for t_point, style in zip(fixed_t, line_styles):
        # 根据固定的 t 点，在 t 数组中找到最接近的索引
        t_idx = np.argmin(np.abs(t - t_point))

        # 提取真实数据下的 Re(x) 与 |q|
        u_fixed_t_exact = u_exact[:, t_idx]  # 固定t下真实的Re(x)在y方向上的分布
        qq_fixed_t_exact = qq_exact_modulus[:, t_idx]  # 固定t下真实的|q|在y方向上的分布
        # 绘制真实数据的曲线
        plt.plot(u_fixed_t_exact, qq_fixed_t_exact, label=f"Real t={t_point}", linestyle=style)

        # 同样提取预测数据下的 Re(x) 与 |q|
        u_fixed_t_pred = u_pred[:, t_idx]
        qq_fixed_t_pred = qq_pred_modulus[:, t_idx]

        # 绘制预测数据的曲线 (使用不同颜色或标记区分)
        plt.plot(u_fixed_t_pred, qq_fixed_t_pred, label=f"Pred t={t_point}", linestyle=style, marker='*', markevery=15)

    plt.xlabel('Re(x)')
    plt.ylabel('|q|')
    # plt.title('在固定 t 下，真实值与预测值的 Re(x) vs |q| 比较')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"LS_Sep/{file_name}_rex_q.png")
    plt.close()

    # #### 绘制模的热图
    plt.figure(figsize=(12, 6))

    # Re(x) 模的原始值
    plt.subplot(2, 2, 1)
    plt.imshow(u_exact, cmap='viridis', extent=[t.min(), t.max(), y.min(), y.max()], aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Re(x) Exact')

    # Re(x) 模的预测值
    plt.subplot(2, 2, 2)
    plt.imshow(u_pred, cmap='viridis', extent=[t.min(), t.max(), y.min(), y.max()], aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Re(x) Predicted')

    # |qq| 模的原始值
    plt.subplot(2, 2, 3)
    plt.imshow(qq_exact_modulus, cmap='viridis', extent=[t.min(), t.max(), y.min(), y.max()], aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('|qq| Exact')

    # |qq| 模的预测值
    plt.subplot(2, 2, 4)
    plt.imshow(qq_pred_modulus, cmap='viridis', extent=[t.min(), t.max(), y.min(), y.max()], aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('|qq| Predicted')

    plt.tight_layout()
    plt.savefig(f"LS_Sep/{file_name}_heatmap.png")
    plt.close()