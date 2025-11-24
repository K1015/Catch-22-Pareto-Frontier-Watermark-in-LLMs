import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Load data from .npz file
# ==========================
data = np.load('dipper_data.npz')
T = int(data['T'])
epsilon = float(data['epsilon'])
change_prob = data['change_prob']
same_prob = data['same_prob']

print(f"Loaded data: T={T}, epsilon={epsilon}")
print(f"Mean change probability: {change_prob.mean():.4f}")

# ==========================
# Plot
# ==========================
x = np.arange(T)
width = 0.45

plt.figure(figsize=(16, 4))  # wide and short

# Bars
plt.bar(x - width/2, same_prob, width,
        label="Token unchanged", color="tab:blue")
plt.bar(x + width/2, change_prob, width,
        label="Token changed",   color="tab:orange")

# Horizontal ε = 0.25 reference line
plt.axhline(y=0.25, color="black", linestyle="--", linewidth=1)

# Axis labels with large font size
plt.xlabel("Token index", fontsize=20)
plt.ylabel("Probability", fontsize=20)

# Larger tick font sizes
plt.xticks(np.arange(0, T, 10),  # every 10 tokens
           np.arange(1, T+1, 10),
           fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0.0, 1.0)

# Legend at the top with larger font
plt.legend(loc="upper center",
           bbox_to_anchor=(0.5, 1.24),
           ncol=2,
           fontsize=18)

plt.tight_layout()
plt.savefig('DIPPER-prob.png', dpi=300, bbox_inches='tight')
plt.show()