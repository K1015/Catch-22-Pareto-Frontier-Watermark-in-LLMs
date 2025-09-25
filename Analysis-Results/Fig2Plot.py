"""
Plotting script for Theorem 3 robustness analysis.
Loads scatter point data from .npy files and generates plots with scatter points only.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set font to Calibri and update plot parameters
try:
    plt.rcParams.update({
        "font.family": "Calibri",
        "font.size": 15,         # tick labels and default text
        "axes.labelsize": 16,    # axis labels
        "axes.titlesize": 16,
        "legend.fontsize": 14
    })
except:
    # Fallback to default sans-serif if Calibri not available
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 14
    })

# Load data from .npz file (preferred method)
try:
    data = np.load('theorem3_robustness_data.npz')
    scatter_epsilon = data['scatter_epsilon']
    scatter_dist_preserving = data['scatter_dist_preserving']
    scatter_bias_free = data['scatter_bias_free']
    scatter_biased = data['scatter_biased']
    knee_bias_free = data['knee_bias_free']
    width_bias_free = data['width_bias_free']
    knee_biased = data['knee_biased']
    width_biased = data['width_biased']
    epsilon_max = data['epsilon_max']
    print("Loaded data from theorem3_robustness_data.npz")
except:
    # Alternative: Load from individual .npy files
    scatter_epsilon = np.load('scatter_epsilon.npy')
    scatter_dist_preserving = np.load('scatter_dist_preserving_robustness.npy')
    scatter_bias_free = np.load('scatter_bias_free_robustness.npy')
    scatter_biased = np.load('scatter_biased_robustness.npy')
    params = np.load('robustness_params.npy')
    knee_bias_free = params[0]
    width_bias_free = params[1]
    knee_biased = params[2]
    width_biased = params[3]
    epsilon_max = params[4]
    print("Loaded data from individual .npy files")

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))

# Plot scatter points only (no lines)
ax.scatter(scatter_epsilon, scatter_dist_preserving, color='green', alpha=1, s=30, 
           edgecolors='none', label='Dist-preserving', zorder=5)
ax.scatter(scatter_epsilon, scatter_bias_free, color='blue', alpha=1, s=30, 
           edgecolors='none', label='Bias-free', zorder=5)
ax.scatter(scatter_epsilon, scatter_biased, color='red', alpha=1, s=30, 
           edgecolors='none', label='Biased', zorder=5)

# Add horizontal line at AUROC = 0.5 (random detection)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, linewidth=1.2)

# Labels and formatting
ax.set_xlabel('Noise Level $\\varepsilon$')
ax.set_ylabel('Robustness (AUROC)')

# Set axis limits
ax.set_xlim([0, epsilon_max])
ax.set_ylim([0.45, 1.05])

# Grid
ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.5)

# Legend
ax.legend(loc='best', framealpha=0.9, markerscale=1.2)

# Ensure clipping
ax.set_clip_on(True)

# Tight layout
plt.tight_layout()

# Save and show
plt.savefig('theorem3_robustness_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
