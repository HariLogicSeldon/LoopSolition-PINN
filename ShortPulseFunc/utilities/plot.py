from pinnstorch.utils.plotting import *
import os

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

    # Extract predictions
    u_pred = preds["u"].view(256, 256).numpy()  # Re(x) prediction
    v_pred = preds["v"].view(256, 256).numpy()  # Im(x) prediction
    p_pred = preds["p"].view(256, 256).numpy()  # Re(q) prediction
    q_pred = preds["q"].view(256, 256).numpy()  # Im(q) prediction
    
    # Calculate predicted modulus
    qq_pred_modulus = np.sqrt(p_pred**2 + q_pred**2)

    # Extract exact solutions
    u_exact = np.array(mesh.solution["u"])
    v_exact = np.array(mesh.solution["v"])
    p_exact = np.array(mesh.solution["p"])
    q_exact = np.array(mesh.solution["q"])
    
    # Calculate exact modulus
    qq_exact_modulus = np.sqrt(p_exact**2 + q_exact**2)

    # Plot comparisons at fixed time points
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

    # Plot heatmap comparisons
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
####3.Error_heatMap

    # Calculate errors
    u_error = np.abs(u_pred - u_exact)
    qq_error = np.abs(qq_pred_modulus - qq_exact_modulus)

    # Plot error heatmaps
    plt.figure(figsize=(12, 6))

    # Re(x) error heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(u_error, cmap='viridis',
               extent=[t.min(), t.max(), y.min(), y.max()],
               aspect='auto', origin='lower',
               vmin=0, vmax=0.5)
    plt.colorbar()
    plt.title('Re(x) Error')
    plt.xlabel('Time')
    plt.ylabel('Space')

    # |q| error heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(qq_error, cmap='viridis',
               extent=[t.min(), t.max(), y.min(), y.max()],
               aspect='auto', origin='lower',
               vmin=0, vmax=0.5)
    plt.colorbar()
    plt.title('|qq| Error')
    plt.xlabel('Time')
    plt.ylabel('Space')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_heatmap.pdf"))
    plt.close()