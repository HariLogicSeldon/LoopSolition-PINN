import matplotlib.pyplot as plt
import numpy as np
from pinnstorch.utils.plotting import *
import os
import pandas as pd
import matplotlib.ticker as mticker
def plot_loop_solition(mesh, preds, logger=None):
    """Plot comparison between predicted and exact solutions for Loop Soliton
    
    Args:
        mesh: Mesh object containing exact solutions
        preds: Dictionary containing predicted results (u, v, p, q)
        logger: Logger object for saving images
    """
    
    # Set output directory
    if logger is not None and hasattr(logger, "log_dir"):
        save_dir = logger.log_dir
    else:
        save_dir = './out'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Extract mesh data
    y = mesh.spatial_domain[0].flatten()  # Spatial coordinates
    t = mesh.time_domain.flatten()  # Time coordinates

    # Determine the new shape based on y and t dimensions
    new_shape = (len(y), len(t))

    # Extract predictions and reshape them
    u_pred = preds["u"].view(new_shape).numpy()  # Re(x) prediction
    v_pred = preds["v"].view(new_shape).numpy()  # Im(x) prediction
    p_pred = preds["p"].view(new_shape).numpy()  # Re(q) prediction
    q_pred = preds["q"].view(new_shape).numpy()  # Im(q) prediction
    
    # Calculate predicted modulus
    qq_pred_modulus = np.sqrt(p_pred**2 + q_pred**2)

    # Extract exact solutions
    u_exact = np.array(mesh.solution["u"])
    v_exact = np.array(mesh.solution["v"])
    p_exact = np.array(mesh.solution["p"])
    q_exact = np.array(mesh.solution["q"])
    
    # Calculate exact modulus
    qq_exact_modulus = np.sqrt(p_exact**2 + q_exact**2)

    # Create a DataFrame
    data = {
        'y': np.tile(y, len(t)),
        't': np.repeat(t, len(y)),
        'u_pred': u_pred.flatten(),
        'v_pred': v_pred.flatten(),
        'p_pred': p_pred.flatten(),
        'q_pred': q_pred.flatten(),
        'qq_pred_modulus': qq_pred_modulus.flatten(),
        'u_exact': u_exact.flatten(),
        'v_exact': v_exact.flatten(),
        'p_exact': p_exact.flatten(),
        'q_exact': q_exact.flatten(),
        'qq_exact_modulus': qq_exact_modulus.flatten(),
    }

    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = os.path.join(save_dir, "solution_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
######## Plot comparisons at fixed time points
    fixed_t = [-1.5, 0, 1.5]
    line_styles_exact = ['-']  # 实线用于真解
    line_styles_pred = ['--']  # 虚线用于模拟解
    plt.figure(figsize=(10, 6))

    # Plot exact and predicted values at fixed time points
    for t_point in fixed_t:
        t_idx = np.argmin(np.abs(t - t_point))

        # Plot exact values
        u_fixed_t_exact = u_exact[:, t_idx]
        qq_fixed_t_exact = qq_exact_modulus[:, t_idx]
        plt.plot(u_fixed_t_exact, qq_fixed_t_exact,
                 label=f"Exact t={t_point}", linestyle=line_styles_exact[0])

        # Plot predicted values
        u_fixed_t_pred = u_pred[:, t_idx]
        qq_fixed_t_pred = qq_pred_modulus[:, t_idx]
        plt.plot(u_fixed_t_pred, qq_fixed_t_pred,
                 label=f"Pred t={t_point}", linestyle=line_styles_pred[0],
                 marker='*', markevery=15)

    # Set plot properties and save
    plt.xlabel('Re(x)')
    plt.ylabel('|q|')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "rex_q.pdf"))
    plt.close()

##### Plot heatmap comparisons
    plt.figure(figsize=(12, 6))
    
    # Exact Re(x) heatmap
    plt.subplot(2, 2, 1)
    plt.imshow(u_exact, cmap='viridis', 
              extent=[t.min(), t.max(), y.min(), y.max()], 
              aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Re(x) Exact')

    # Predicted Re(x) heatmap
    plt.subplot(2, 2, 2)
    plt.imshow(u_pred, cmap='viridis', 
              extent=[t.min(), t.max(), y.min(), y.max()], 
              aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Re(x) Predicted')

    # Exact |q| heatmap
    plt.subplot(2, 2, 3)
    plt.imshow(qq_exact_modulus, cmap='viridis', 
              extent=[t.min(), t.max(), y.min(), y.max()], 
              aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('|qq| Exact')

    # Predicted |q| heatmap
    plt.subplot(2, 2, 4)
    plt.imshow(qq_pred_modulus, cmap='viridis', 
              extent=[t.min(), t.max(), y.min(), y.max()], 
              aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('|qq| Predicted')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "heatmap.pdf"))
    plt.close()
####3.三维图像
    # 创建一个三维坐标轴对象
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Meshgrid 生成网格数据
    T, Y = np.meshgrid(t, y)
    QQ = qq_exact_modulus  # |q| 已经是计算好的数据
    QQ_pred = qq_pred_modulus
    # print(u_pred.shape,T.shape,Y.shape,QQ.shape,QQ_pred.shape)

    # 绘制 3D 图像
    ax.plot_wireframe(T, u_exact, QQ, color='red', linestyle='-', linewidth=0.7)
    surf = ax.plot_surface(T, u_pred, QQ_pred, cmap='viridis', edgecolor='None',alpha=0.7)
    # 设置三维图像的角度
    ax.view_init(elev=30, azim=45)  # 仰角30度，方位角45度

    # 添加标题和标签
    ax.set_title("3D Surface Plot of |q| - Re(x) - t")
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Space (x)')
    ax.set_zlabel('|q|')

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='red', linestyle='-', linewidth=0.7, label='Exact |q|'),
        plt.Line2D([0], [0], color='purple', marker='o', markersize=8, linestyle='None', label='Predicted |q|')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(save_dir, "q_3d_plot.pdf"))
    plt.show()
    # 保存图片
    plt.close()


