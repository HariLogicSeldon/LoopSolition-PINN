import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Define functions for complex solutions
def xx(y, t):
    """Calculate the complex solution x(y,t)
    
    Args:
        y: Spatial coordinate
        t: Time coordinate
    
    Returns:
        Complex solution x(y,t)
    """
    # Numerator
    numerator = (
        72j*np.exp(-0.1j*(-20*y + t))
        + (-11 - 3j - 15*y)*np.exp(-10*y - (11*t)/30.)
        + (11 - 3j - 15*y)*np.exp(10*y + (11*t)/30.)
        + (13 - 39j - 195*y)*np.exp(-2*y + t/30.)
        + (-13 - 39j - 195*y)*np.exp(2*y - t/30.)
        + 360*y*np.cos(-2*y + t/10)
    )

    # Denominator
    denominator = (
        -195*np.exp(-2*y + t/30.)
        - 15*np.exp(10*y + (11*t)/30.)
        - 15*np.exp(-10*y - (11*t)/30.)
        - 195*np.exp(2*y - t/30.)
        + 360*np.cos(-2*y + t/10.)
    )

    return numerator / denominator

def qq(y, t):
    """Calculate the complex solution q(y,t)
    
    Args:
        y: Spatial coordinate
        t: Time coordinate
    
    Returns:
        Complex solution q(y,t)
    """
    # Numerator
    numerator = ((-24 + 36j)*np.exp((1/6 - 0.1j)*t + (6 + 2j)*y)
             - (24 + 36j)*np.exp(-(1/6 + 0.1j)*t + (-6 + 2j)*y)
             + (2 + 36j)*np.exp(-4*y - t/5)
            + (2 - 36j)*np.exp(4*y + t/5))

    # Denominator
    denominator = (-195*np.exp(-2*y + t/30)
               - 15*np.exp(10*y + (11*t)/30)
               - 15*np.exp(-10*y - (11*t)/30)
               - 195*np.exp(2*y - t/30)
               + 360*np.cos(-2*y + t/10))
    
    return numerator / denominator

# Generate grid data
y = np.linspace(-1.5,1.5, 256)  # Spatial coordinate range [-1.5, 1.5]
t = np.linspace(-11,11, 256)  # Time coordinate range [-1.5, 1.5]
Y, T = np.meshgrid(y, t)

# Calculate analytical solutions
XX = xx(Y, T)  # Calculate complex solution x(y,t)
QQ = qq(Y, T)  # Calculate complex solution q(y,t)


# Prepare data for saving
output_data = {
    'y': y,
    't': t,
    'xx': XX,
    'qq': QQ
}

# Set data save path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
data_dir = os.path.join(root_dir, 'data')

# Create data directory
os.makedirs(data_dir, exist_ok=True)
output_path = os.path.join(data_dir, 'MultiSolition.mat')
scipy.io.savemat(output_path, output_data)

# Load saved data for visualization
data = scipy.io.loadmat(output_path)
y = data['y'].flatten()
t = data['t'].flatten()
xx_data = data['xx'].T
qq_data = data['qq'].T

# Create visualization grid
T, Y = np.meshgrid(t, y)
XX_plot = xx_data.real  # Extract real part
QQ_plot = np.abs(qq_data)  # Calculate modulus

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D surface
surface = ax.plot_surface(T, XX_plot, QQ_plot, cmap='plasma',
                         edgecolor='none', alpha=0.8)

# Add contour projection
contour = ax.contourf(T, XX_plot, QQ_plot, zdir='z',
                      offset=np.min(QQ_plot) - 0.1,
                      cmap='plasma', alpha=0.8)

# Set plot properties
cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('|q|', fontsize=12)
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('Re(X)', fontsize=12)
ax.set_zlabel('|q|', fontsize=12)
ax.set_title('3D Plot of Re(x) and |q| with t and Heatmap Projection',
             fontsize=14)
ax.set_zlim(np.min(QQ_plot) - 0.1, np.max(QQ_plot))

# Plot 2D slices
fixed_t_indices = [0, len(t)//2, -1]  # Select time slice points
line_styles = ['--', '-.', ':']
plt.figure(figsize=(10, 6))

# Plot slices at different time points
for t_idx, style in zip(fixed_t_indices, line_styles):
    t_value = t[t_idx]
    x_fixed_t = xx_data[:, t_idx].real
    q_fixed_t = np.abs(qq_data[:, t_idx])
    plt.plot(x_fixed_t, q_fixed_t,
             label=f"t={t_value:.2f}",
             linestyle=style)

# Set 2D plot properties
plt.xlabel('Re(x)')
plt.ylabel('|q|')
plt.title('2D Plot of Re(x) vs |q| at Fixed t and Varying y')
plt.legend()
plt.grid(True)
plt.show()
