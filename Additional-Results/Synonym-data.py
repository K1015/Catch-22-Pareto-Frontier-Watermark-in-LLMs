import numpy as np

# ==========================
# Generate Synonym-Substitution-like pattern
# ==========================
T = 100
epsilon = 0.18    # synonym substitution tends to cause higher edit rates
rng = np.random.default_rng(1)
positions = np.arange(T)

# Start from a flat baseline
change_prob = np.full(T, epsilon)

# Stronger non-uniformity → many Gaussians with wider sigma
num_gaussians = 30
for _ in range(num_gaussians):
    center = rng.integers(0, T)
    sigma  = rng.uniform(8, 20)           # wider than DIPPER
    height = rng.uniform(-0.10, 0.18)     # synonym substitution more aggressive
    gaussian = height * np.exp(-0.5 * ((positions - center) / sigma) ** 2)
    change_prob += gaussian

# Heavy iid noise (synonym replacement is more chaotic)
noise = rng.normal(loc=0.0, scale=0.12, size=T)
change_prob += noise

# Positional ramp: synonym-based attacks often replace tokens later in the text
ramp = np.linspace(0.0, 0.15, T)   # increases change in 51–100 region
change_prob += ramp

# Slight smoothing to avoid sharp spikes
change_prob = 0.7 * change_prob + 0.3 * np.roll(change_prob, 1)

# Normalize to target mean epsilon and clip to realistic range
change_prob *= (epsilon / change_prob.mean())
change_prob = np.clip(change_prob, 0.05, 0.55)   # synonym substitution can be up to 50%

same_prob = 1.0 - change_prob

print("Mean change probability:", change_prob.mean())

# ==========================
# Save to .npz file
# ==========================
np.savez(
    'synonym_substitution_data.npz',
    T=T,
    epsilon=epsilon,
    change_prob=change_prob,
    same_prob=same_prob
)

print("Data saved to synonym_substitution_data.npz")