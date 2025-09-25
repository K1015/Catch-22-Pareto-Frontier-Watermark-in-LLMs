"""
Plotting script for Pareto frontier analysis.
Loads scatter point data from .npz files and generates plots showing trade-offs.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set font with fallback options
try:
    plt.rcParams.update({
        "font.family": "Calibri",
        "font.size": 15,
        "axes.labelsize": 16,
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

# Load data from .npz file
try:
    data = np.load('pareto_frontier_data.npz')
    
    # Regime 1 (Low noise)
    biased_detect_01 = data['biased_detect_01']
    biased_robust_01 = data['biased_robust_01']
    bias_free_detect_01 = data['bias_free_detect_01']
    bias_free_robust_01 = data['bias_free_robust_01']
    dist_detect_01 = data['dist_detect_01']
    dist_robust_01 = data['dist_robust_01']
    
    # Regime 2 (Medium noise)
    biased_detect_02 = data['biased_detect_02']
    biased_robust_02 = data['biased_robust_02']
    bias_free_detect_02 = data['bias_free_detect_02']
    bias_free_robust_02 = data['bias_free_robust_02']
    dist_detect_02 = data['dist_detect_02']
    dist_robust_02 = data['dist_robust_02']
    
    # Regime 3 (High noise)
    biased_detect_03 = data['biased_detect_03']
    biased_robust_03 = data['biased_robust_03']
    bias_free_detect_03 = data['bias_free_detect_03']
    bias_free_robust_03 = data['bias_free_robust_03']
    dist_detect_03 = data['dist_detect_03']
    dist_robust_03 = data['dist_robust_03']
    
    # Optimal points
    optimal_regime1_detect = data['optimal_regime1_detect']
    optimal_regime1_robust = data['optimal_regime1_robust']
    optimal_regime2_detect = data['optimal_regime2_detect']
    optimal_regime2_robust = data['optimal_regime2_robust']
    optimal_regime3_detect = data['optimal_regime3_detect']
    optimal_regime3_robust = data['optimal_regime3_robust']
    
    n_points = data['n_points']
    print("Loaded data from pareto_frontier_data.npz")
    
except:
    # Alternative: Load from individual regime files
    regime1 = np.load('pareto_regime1.npz')
    regime2 = np.load('pareto_regime2.npz')
    regime3 = np.load('pareto_regime3.npz')
    
    # Extract data for each regime
    biased_detect_01 = regime1['biased_detect']
    biased_robust_01 = regime1['biased_robust']
    bias_free_detect_01 = regime1['bias_free_detect']
    bias_free_robust_01 = regime1['bias_free_robust']
    dist_detect_01 = regime1['dist_detect']
    dist_robust_01 = regime1['dist_robust']
    
    biased_detect_02 = regime2['biased_detect']
    biased_robust_02 = regime2['biased_robust']
    bias_free_detect_02 = regime2['bias_free_detect']
    bias_free_robust_02 = regime2['bias_free_robust']
    dist_detect_02 = regime2['dist_detect']
    dist_robust_02 = regime2['dist_robust']
    
    biased_detect_03 = regime3['biased_detect']
    biased_robust_03 = regime3['biased_robust']
    bias_free_detect_03 = regime3['bias_free_detect']
    bias_free_robust_03 = regime3['bias_free_robust']
    dist_detect_03 = regime3['dist_detect']
    dist_robust_03 = regime3['dist_robust']
    
    # Default optimal points if not in separate files
    optimal_regime1_detect = 0.015
    optimal_regime1_robust = 0.98
    optimal_regime2_detect = 0.14
    optimal_regime2_robust = 0.85
    optimal_regime3_detect = 0.24
    optimal_regime3_robust = 0.66
    
    n_points = 5
    print("Loaded data from individual regime files")

# Create figure
fig, ax = plt.subplots(figsize=(6, 5.5))

# ---------- Plot points for each regime ----------
# Regime 1 (ε < 0.11) - red
ax.scatter(biased_detect_01, biased_robust_01, 
           s=60, alpha=0.7, color='red', marker='^', label='_nolegend_')
ax.scatter(bias_free_detect_01, bias_free_robust_01, 
           s=60, alpha=0.7, color='red', marker='s', label='_nolegend_')
ax.scatter(dist_detect_01, dist_robust_01, 
           s=60, alpha=0.7, color='red', marker='o', label='_nolegend_')

# Regime 2 (0.11 ≤ ε < 0.17) - blue
ax.scatter(biased_detect_02, biased_robust_02, 
           s=60, alpha=0.7, color='blue', marker='^', label='_nolegend_')
ax.scatter(bias_free_detect_02, bias_free_robust_02, 
           s=60, alpha=0.7, color='blue', marker='s', label='_nolegend_')
ax.scatter(dist_detect_02, dist_robust_02, 
           s=60, alpha=0.7, color='blue', marker='o', label='_nolegend_')

# Regime 3 (0.17 ≤ ε < 0.32) - green
ax.scatter(biased_detect_03, biased_robust_03, 
           s=60, alpha=0.7, color='green', marker='^', label='_nolegend_')
ax.scatter(bias_free_detect_03, bias_free_robust_03, 
           s=60, alpha=0.7, color='green', marker='s', label='_nolegend_')
ax.scatter(dist_detect_03, dist_robust_03, 
           s=60, alpha=0.7, color='green', marker='o', label='_nolegend_')

# ---------- Add optimal watermarking points for each regime ----------
# Optimal for Regime 1 (Dist-preserving is optimal)
ax.scatter([optimal_regime1_detect], [optimal_regime1_robust], s=400, marker='*', color='gold', 
           edgecolors='black', linewidth=2, zorder=11, label='Optimal')

# Optimal for Regime 2 (Bias-free is optimal)
ax.scatter([optimal_regime2_detect], [optimal_regime2_robust], s=400, marker='*', color='gold', 
           edgecolors='black', linewidth=2, zorder=11, label='_nolegend_')

# Optimal for Regime 3 (Biased is optimal)
ax.scatter([optimal_regime3_detect], [optimal_regime3_robust], s=400, marker='*', color='gold', 
           edgecolors='black', linewidth=2, zorder=11, label='_nolegend_')

# ---------- Create legend for markers and colors ----------
# Dummy plots for legend
from matplotlib.lines import Line2D

# Method markers
legend_elements_methods = [
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Biased'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='Bias-free'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Dist-preserving'),
]

# Noise regime colors
legend_elements_regimes = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='ε < 0.005'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='0.12 ≤ ε < 0.15'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label=' ε > 0.15'),
]

# Add legends
legend1 = ax.legend(handles=legend_elements_methods, loc='upper right', title='Method', framealpha=0.9)
legend2 = ax.legend(handles=legend_elements_regimes, loc='lower left', title='Noise Level', framealpha=0.9)
ax.add_artist(legend1)  # Add first legend back

# ---------- Formatting ----------
ax.set_xlabel('Detectability (TV Distance)')
ax.set_ylabel('Robustness (AUROC)')

# Set axis limits - TV Distance capped at 0.3
ax.set_xlim([0, 0.3])
ax.set_ylim([0.48, 1.02])

# Add horizontal line at AUROC = 0.5
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, linewidth=1.2)

# Grid
ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.5)

# Ensure clipping
ax.set_clip_on(True)

# Tight layout
plt.tight_layout()

# Save and show
plt.savefig('pareto_frontier_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
