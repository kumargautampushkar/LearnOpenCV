import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------------------------------------------------------
# 1. Setup Simulation Domain
# -----------------------------------------------------------------------------
points = 1000
limit = 50
x = np.linspace(-limit, limit, points)
dx = x[1] - x[0]

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------

def get_blob(size, x_domain):
    """Generates a 1D rectangular pulse (boxcar) centered at 0."""
    y = np.zeros_like(x_domain)
    mask = np.abs(x_domain) <= (size / 2.0)
    y[mask] = 1.0
    return y

def get_kernels(sigma, dx):
    """
    Generates 4 distinct kernels:
    1. Gaussian (G)
    2. 1st Derivative of Gaussian (dG/dx)
    3. 2nd Derivative of Gaussian (d^2G/dx^2)
    4. Sigma-Normalized LoG (sigma^2 * d^2G/dx^2)
    """
    # Create kernel domain 
    # (tails need to be wide enough to capture the function decay)
    radius = int(4.0 * sigma) + 1
    k_x = np.arange(-radius, radius + dx, dx)
    
    # --- 1. Gaussian ---
    norm = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    g = norm * np.exp(-(k_x**2) / (2 * sigma**2))
    
    # --- 2. 1st Derivative (dG/dx) ---
    # Formula: -x/sigma^2 * G
    g_d1 = -(k_x / sigma**2) * g
    
    # --- 3. 2nd Derivative (d^2G/dx^2) ---
    # Formula: (x^2/sigma^4 - 1/sigma^2) * G
    g_d2 = ((k_x**2 / sigma**4) - (1 / sigma**2)) * g
    
    # --- 4. Normalized LoG ---
    # Formula: sigma^2 * (d^2G/dx^2)
    norm_log = (sigma**2) * g_d2
    
    return g, g_d1, g_d2, norm_log

# -----------------------------------------------------------------------------
# 3. Initialize Plots
# -----------------------------------------------------------------------------
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
plt.subplots_adjust(left=0.15, bottom=0.15, top=0.96, hspace=0.4)

init_sigma = 2.0
init_size = 10.0

# Initial calculations
blob = get_blob(init_size, x)
k_g, k_d1, k_d2, k_nlog = get_kernels(init_sigma, dx)

# Convolutions (multiply by dx for integral approximation)
c_g = np.convolve(blob, k_g, mode='same') * dx
c_d1 = np.convolve(blob, k_d1, mode='same') * dx
c_d2 = np.convolve(blob, k_d2, mode='same') * dx
c_nlog = np.convolve(blob, k_nlog, mode='same') * dx

# --- Plot 1: Gaussian ---
l_b1, = ax1.plot(x, blob, 'k--', alpha=0.3, label='Input Blob')
l_g, = ax1.plot(x, c_g, 'b-', linewidth=2, label='Conv with Gaussian')
ax1.set_title("1. Gaussian (Smoothing)")
ax1.legend(loc='upper right', fontsize='small')
ax1.grid(True, alpha=0.3)

# --- Plot 2: 1st Derivative ---
l_b2, = ax2.plot(x, blob, 'k--', alpha=0.3)
l_d1, = ax2.plot(x, c_d1, 'g-', linewidth=2, label='Conv with dG/dx')
ax2.set_title("2. First Derivative (Edge Detection)")
ax2.legend(loc='upper right', fontsize='small')
ax2.grid(True, alpha=0.3)

# --- Plot 3: 2nd Derivative (Raw) ---
l_b3, = ax3.plot(x, blob, 'k--', alpha=0.3)
l_d2, = ax3.plot(x, c_d2, 'r-', linewidth=1.5, label='Conv with d^2G/dx^2')
ax3.set_title("3. Second Derivative (Raw LoG)")
ax3.legend(loc='upper right', fontsize='small')
ax3.grid(True, alpha=0.3)

# --- Plot 4: Normalized LoG ---
l_b4, = ax4.plot(x, blob, 'k--', alpha=0.3)
l_nlog, = ax4.plot(x, c_nlog, 'm-', linewidth=2, label='Conv with Norm LoG')
ax4.set_title("4. Normalized LoG (Scale Selection)")
ax4.set_xlabel("Position (x)")
ax4.legend(loc='upper right', fontsize='small')
ax4.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# 4. Sliders and Update Logic
# -----------------------------------------------------------------------------
ax_sigma = plt.axes([0.2, 0.06, 0.6, 0.02])
ax_size = plt.axes([0.2, 0.03, 0.6, 0.02])

slider_sigma = Slider(ax_sigma, 'Sigma', 0.1, 15.0, valinit=init_sigma)
slider_size = Slider(ax_size, 'Blob Size', 1.0, 40.0, valinit=init_size)

def update(val):
    sig = slider_sigma.val
    sz = slider_size.val
    
    # Update Blob
    new_blob = get_blob(sz, x)
    l_b1.set_ydata(new_blob)
    l_b2.set_ydata(new_blob)
    l_b3.set_ydata(new_blob)
    l_b4.set_ydata(new_blob)
    
    # Update Kernels
    kg, kd1, kd2, knlog = get_kernels(sig, dx)
    
    # Update Convolutions
    res_g = np.convolve(new_blob, kg, mode='same') * dx
    res_d1 = np.convolve(new_blob, kd1, mode='same') * dx
    res_d2 = np.convolve(new_blob, kd2, mode='same') * dx
    res_nlog = np.convolve(new_blob, knlog, mode='same') * dx
    
    # Update Lines
    l_g.set_ydata(res_g)
    l_d1.set_ydata(res_d1)
    l_d2.set_ydata(res_d2)
    l_nlog.set_ydata(res_nlog)
    
    # Rescale axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.relim()
        ax.autoscale_view()
        
    fig.canvas.draw_idle()

slider_sigma.on_changed(update)
slider_size.on_changed(update)

plt.show()