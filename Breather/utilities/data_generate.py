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

    sqrt5 = np.sqrt(5)
    term1_exp = np.exp(18 * ((-1 + 1j) * t - (1 + 1j) * y) * (sqrt5 + 20 / 9) / (45 + 20 * sqrt5))
    term2_exp = np.exp(-18 * (sqrt5 + 20 / 9) * ((1 + 1j) * t + (1 - 1j) * y) / (45 + 20 * sqrt5))
    term3_exp = np.exp(-4 * (9 * sqrt5 + 20) * (t + y) / (45 + 20 * sqrt5))
    
    numerator = (

            ((-18j * t + 80j * y + 200j) * sqrt5 + 45 * t - 200 * y + 400) * term1_exp +

            ((-200j - 80j * y + 18j * t) * sqrt5 + 45 * t - 200 * y + 400) * term2_exp +

            ((-162 * t + 720 * y + 1800) * sqrt5 - 405 * t + 1800 * y + 3600) * term3_exp +

            (162 * t - 720 * y - 1800) * sqrt5 - 405 * t + 1800 * y + 3600

    )
    denominator = (

            (100j * sqrt5 - 250) * term1_exp - 100j * sqrt5 * term2_exp +

            900 * term3_exp * sqrt5 - 900 * sqrt5 + 2250 * term3_exp - 250 * term2_exp + 2250

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
    sqrt5 = np.sqrt(5)
    factor1 = (9 + 4 * sqrt5) ** 2
    factor2 = (45 + 20 * sqrt5)

    # Exponential terms
    exp1 = np.exp((( - (644 + 288j) * t + (-644 + 360j) * y) * sqrt5 - (1440 + 644j) * t + (-1440 + 805j) * y) / (5 * factor1))
    exp2 = np.exp((( - (322 + 610j) * t + (-322 + 682j) * y) * sqrt5 - (720 + 1364j) * t + (-720 + 1525j) * y) / (5 * factor1))
    exp3 = np.exp((( - (322 + 34j) * t + (-322 + 38j) * y) * sqrt5 - (720 + 76j) * t + (-720 + 85j) * y) / (5 * factor1))
    exp4 = np.exp((( - (36 + 16j) * t + (-36 + 20j) * y) * sqrt5 - (80 + 36j) * t + (-80 + 45j) * y) / factor2)
    exp5 = np.exp((( - (18 + 34j) * t + (-18 + 38j) * y) * sqrt5 - (40 + 76j) * t + (-40 + 85j) * y) / factor2)
    exp6 = np.exp((( - (18 + 2j) * t + (-18 + 2j) * y) * sqrt5 - (40 + 4j) * t + (-40 + 5j) * y) / factor2)
    exp7 = np.exp((-1j / 5) * (-5 * y + 4 * t))

    # Numerator terms
    term1 = (-150 + 120j + (-60 + 60j) * sqrt5) * exp1
    term2 = (150 - 120j + (-60 + 60j) * sqrt5) * exp2
    term3 = (150 - 120j + (60 - 60j) * sqrt5) * exp3
    term4 = (54 * sqrt5 + 135) * exp4
    term5 = (-6j * sqrt5 - 15) * exp5
    term6 = (6j * sqrt5 - 15) * exp6
    term7 = -60 * exp7 * (1/4 - 2j + (-1/10 + 1j) * sqrt5)

    numerator = term1 + term2 + term3 + term4 + term5 + term6 + term7

    # Denominator terms
    exp8 = np.exp(18 * ((-1 + 1j) * t - (1 + 1j) * y) * (sqrt5 + 20 / 9) / (45 + 20 * sqrt5))
    exp9 = np.exp(-18 * (sqrt5 + 20 / 9) * ((1 + 1j) * t + (1 - 1j) * y) / (45 + 20 * sqrt5))
    exp10 = np.exp(-4 * (9 * sqrt5 + 20) * (t + y) / (45 + 20 * sqrt5))

    term8 = (10j * sqrt5 - 25) * exp8
    term9 = -10j * sqrt5 * exp9
    term10 = 90 * exp10 * sqrt5
    term11 = -90 * sqrt5
    term12 = 225 * exp10
    term13 = -25 * exp9
    term14 = 225

    denominator = (term8 + term9 + term10 + term11 + term12 + term13 + term14)

    return numerator / denominator


# Generate grid data
y = np.linspace(-6,6, 1024)  # Spatial coordinate range [-1.5, 1.5]
t = np.linspace(-1.5,1.5, 256)  # Time coordinate range [-1.5, 1.5]
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
output_path = os.path.join(data_dir, 'Breather.mat')
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
