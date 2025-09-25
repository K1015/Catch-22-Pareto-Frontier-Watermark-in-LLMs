"""
Plotting script for Theorem 1 detectability analysis.
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
    data = np.load('theorem1_detectability_data.npz')
    scatter_L = data['scatter_L']
    scatter_greedy = data['scatter_greedy']
    scatter_biased = data['scatter_biased']
    scatter_bias_free = data['scatter_bias_free']
    scatter_dist_preserving = data['scatter_dist_preserving']
    c1_biased = data['c1_biased']
    c2_bias_free = data['c2_bias_free']
    print("Loaded data from theorem1_detectability_data.npz")
except:
    # Alternative: Load from individual .npy files
    scatter_L = np.load('scatter_L.npy')
    scatter_greedy = np.load('scatter_greedy.npy')
    scatter_biased = np.load('scatter_biased.npy')
    scatter_bias_free = np.load('scatter_bias_free.npy')
    scatter_dist_preserving = np.load('scatter_dist_preserving.npy')
    params = np.load('model_params.npy')
    c1_biased = params[0]
    c2_bias_free = params[1]
    print("Loaded data from individual .npy files")

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))

# Plot scatter points only (no lines)
ax.scatter(scatter_L, scatter_greedy, color='purple', alpha=1, s=30, 
           edgecolors='none', label='Greedy: $O(1)$')
ax.scatter(scatter_L, scatter_biased, color='blue', alpha=1, s=30, 
           edgecolors='none', label=f'Biased: $O(\sqrt{{T}})$')
ax.scatter(scatter_L, scatter_bias_free, color='red', alpha=1, s=30, 
           edgecolors='none', label=f'Bias-free: $O(\sqrt{{T}})$')
ax.scatter(scatter_L, scatter_dist_preserving, color='green', alpha=1, s=30, 
           edgecolors='none', label='Distribution-preserving: 0')

# Add reference lines
ax.axhline(y=0.7, color='darkgray', linestyle=':', alpha=0.3, linewidth=1.5)
ax.axhline(y=1.0, color='darkgray', linestyle=':', alpha=0.3, linewidth=1.5)

# Set scales
ax.set_xscale('log')
ax.set_yscale('linear')

# Labels and formatting
ax.set_xlabel('Text Length $T$ (tokens)')
ax.set_ylabel('Total Variation Distance')

# Set axis limits
ax.set_xlim([40, 550])
ax.set_ylim([0, 1.05])

# Grid
ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.5)

# Add y-axis ticks
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Legend
ax.legend(loc='best', framealpha=0.9, markerscale=1.2)

# Tight layout
plt.tight_layout()

# Save and show
plt.savefig('theorem1_detectability_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
