import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------------------------------------------------------
# 1. Setup Simulation Domain & Noise
# -----------------------------------------------------------------------------
points = 1000
limit = 50
x = np.linspace(-limit, limit, points)
dx = x[1] - x[0]

# Pre-generate a noise pattern so the slider scales it smoothly
# rather than generating new random numbers every frame (which causes flickering).
np.random.seed(42) # Fixed seed for consistency
noise_pattern = np.random.normal(0, 1, points)

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------

def get_base_blob(size, x_domain):
    """Generates the clean 1D rectangular pulse."""
    y = np.zeros_like(x_domain)
    mask = np.abs(x_domain) <= (size / 2.0)
    y[mask] = 1.0
    return y

def evaluate_kernels_on_grid(sigma, grid):
    """Evaluates analytic kernels on the viewing grid for visualization."""
    norm = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    g = norm * np.exp(-(grid**2) / (2 * sigma**2))
    g_d1 = -(grid / sigma**2) * g
    g_d2 = ((grid**2 / sigma**4) - (1 / sigma**2)) * g
    return g, g_d1, g_d2

def get_convolution_kernels(sigma, dx):
    """Generates compact kernels for calculation."""
    radius = int(4.0 * sigma) + 1
    k_x = np.arange(-radius, radius + dx, dx)
    
    # Reuse the evaluation logic
    g, g_d1, g_d2 = evaluate_kernels_on_grid(sigma, k_x)
    
    # Calculate Normalized LoG kernel
    k_nlog = (sigma**2) * g_d2
    
    return g, g_d1, g_d2, k_nlog

# -----------------------------------------------------------------------------
# 3. Initialize Plots
# -----------------------------------------------------------------------------
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(9, 14), sharex=True)
plt.subplots_adjust(left=0.15, bottom=0.12, top=0.96, hspace=0.4)

# Initial Parameters
init_sigma = 2.0
init_size = 10.0
init_noise = 0.0

# Initial Data Calculation
base_blob = get_base_blob(init_size, x)
blob = base_blob + (init_noise * noise_pattern)

k_g, k_d1, k_d2, k_nlog = get_convolution_kernels(init_sigma, dx)
v_g, v_d1, v_d2 = evaluate_kernels_on_grid(init_sigma, x)

# Convolutions
c_g = np.convolve(blob, k_g, mode='same') * dx
c_d1 = np.convolve(blob, k_d1, mode='same') * dx
c_d2 = np.convolve(blob, k_d2, mode='same') * dx
c_nlog = np.convolve(blob, k_nlog, mode='same') * dx

# --- Plot 1: Gaussian Smoothing ---
l_b1, = ax1.plot(x, blob, 'k--', alpha=0.3, label='Input (Blob + Noise)')
l_res_g, = ax1.plot(x, c_g, 'b-', linewidth=2, label='Result: Gaussian Smooth')
ax1.set_title("1. Result: Gaussian Smoothing (Noise Suppression)")
ax1.legend(loc='upper right', fontsize='x-small')
ax1.grid(True, alpha=0.3)

# --- Plot 2: Kernel Shapes ---
l_k_g, = ax2.plot(x, v_g, 'b:', label='Kernel: Gaussian')
l_k_d1, = ax2.plot(x, v_d1, 'g-', label='Kernel: dG/dx')
l_k_d2, = ax2.plot(x, v_d2, 'r-.', label='Kernel: LoG')
ax2.set_title("2. Visual: The Kernels (Defined by Sigma)")
ax2.legend(loc='upper right', fontsize='x-small')
ax2.grid(True, alpha=0.3)
ax2.set_ylabel("Kernel Amp")

# --- Plot 3: Edge Detection ---
l_b3, = ax3.plot(x, blob, 'k--', alpha=0.3)
l_res_d1, = ax3.plot(x, c_d1, 'g-', linewidth=2, label='Result: dG/dx')
ax3.set_title("3. Result: Edge Detection (1st Deriv)")
ax3.legend(loc='upper right', fontsize='x-small')
ax3.grid(True, alpha=0.3)

# --- Plot 4: Raw LoG ---
l_b4, = ax4.plot(x, blob, 'k--', alpha=0.3)
l_res_d2, = ax4.plot(x, c_d2, 'r-', linewidth=1.5, label='Result: LoG')
ax4.set_title("4. Result: Laplacian (2nd Deriv)")
ax4.legend(loc='upper right', fontsize='x-small')
ax4.grid(True, alpha=0.3)

# --- Plot 5: Norm LoG ---
l_b5, = ax5.plot(x, blob, 'k--', alpha=0.3)
l_res_nlog, = ax5.plot(x, c_nlog, 'm-', linewidth=2, label='Result: Norm LoG')
ax5.set_title("5. Result: Normalized LoG (Scale Invariant)")
ax5.set_xlabel("Position (x)")
ax5.legend(loc='upper right', fontsize='x-small')
ax5.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# 4. Sliders and Update Logic
# -----------------------------------------------------------------------------
# Define axes for sliders [left, bottom, width, height]
ax_sigma = plt.axes([0.2, 0.07, 0.6, 0.02])
ax_size  = plt.axes([0.2, 0.04, 0.6, 0.02])
ax_noise = plt.axes([0.2, 0.01, 0.6, 0.02])

slider_sigma = Slider(ax_sigma, 'Sigma', 0.1, 15.0, valinit=init_sigma)
slider_size  = Slider(ax_size,  'Blob Size', 1.0, 40.0, valinit=init_size)
slider_noise = Slider(ax_noise, 'Noise Lvl', 0.0, 0.5, valinit=init_noise)

def update(val):
    sig = slider_sigma.val
    sz = slider_size.val
    ns = slider_noise.val
    
    # 1. Recreate Blob with Noise
    # Scaling fixed noise pattern ensures smooth transitions
    current_blob = get_base_blob(sz, x) + (ns * noise_pattern)
    
    # Update "Ghost" input lines on all plots
    l_b1.set_ydata(current_blob)
    l_b3.set_ydata(current_blob)
    l_b4.set_ydata(current_blob)
    l_b5.set_ydata(current_blob)
    
    # 2. Update Kernels
    kg, kd1, kd2, knlog = get_convolution_kernels(sig, dx)
    
    # 3. Update Kernel Visualization (Plot 2)
    vg, vd1, vd2 = evaluate_kernels_on_grid(sig, x)
    l_k_g.set_ydata(vg)
    l_k_d1.set_ydata(vd1)
    l_k_d2.set_ydata(vd2)
    
    # 4. Update Convolutions (The heavy lifting)
    res_g = np.convolve(current_blob, kg, mode='same') * dx
    res_d1 = np.convolve(current_blob, kd1, mode='same') * dx
    res_d2 = np.convolve(current_blob, kd2, mode='same') * dx
    res_nlog = np.convolve(current_blob, knlog, mode='same') * dx
    
    # 5. Update Result Lines
    l_res_g.set_ydata(res_g)
    l_res_d1.set_ydata(res_d1)
    l_res_d2.set_ydata(res_d2)
    l_res_nlog.set_ydata(res_nlog)
    
    # 6. Rescale Y-Axes
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.relim()
        ax.autoscale_view()
        
    fig.canvas.draw_idle()

# Register update function
slider_sigma.on_changed(update)
slider_size.on_changed(update)
slider_noise.on_changed(update)

plt.show()