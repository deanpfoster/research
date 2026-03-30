#!/usr/bin/env python3
"""
Demonstrate anisotropic temperature from martingale coupling.
2 agents in 2D: agent 0 pinned at anchor point, agent 1 diffuses under
regularizer + SGD noise (+ coupling if enabled).

4-panel figure: {uncoupled, coupled} × {anchor direction, orthogonal direction}
Shows coupling heats only the anchor axis while orthogonal direction is unchanged.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ─────────────────────────────────────────────────────────────
eta = 0.1
noise_std = 1.0
gamma = 0.3
lam = 0.005

# Anchor at (10, 0): coupling noise is purely along x-axis
anchor = np.array([10.0, 0.0])
anchor_hat = anchor / np.linalg.norm(anchor)  # [1, 0]
orth_hat = np.array([0.0, 1.0])

n_steps = 20000

# Theory predictions for equilibrium variance
D_sgd = (eta * noise_std)**2 / 2       # SGD diffusion coefficient
D_cpl = (eta * gamma * np.linalg.norm(anchor))**2 / 2  # coupling diffusion (bump dir only)
var_theory_unc = D_sgd / (2 * lam * eta)
var_theory_cpl_bump = (D_sgd + D_cpl) / (2 * lam * eta)
var_theory_cpl_orth = D_sgd / (2 * lam * eta)

print(f"D_sgd = {D_sgd:.4f},  D_coupling = {D_cpl:.4f}")
print(f"Theory variance (uncoupled, both dirs): {var_theory_unc:.2f}")
print(f"Theory variance (coupled, bump dir):    {var_theory_cpl_bump:.2f}  ({var_theory_cpl_bump/var_theory_unc:.1f}× boost)")
print(f"Theory variance (coupled, orth dir):    {var_theory_cpl_orth:.2f}  (1.0× = unchanged)")


def run_2d(coupled, seed):
    """Agent diffuses in 2D under regularizer + noise (+ coupling)."""
    rng = np.random.default_rng(seed)
    pos = np.zeros(2)

    proj_bump = np.zeros(n_steps + 1)
    proj_orth = np.zeros(n_steps + 1)

    for t in range(n_steps):
        # Regularizer gradient (ascent on -lam*||x||^2) + SGD noise
        grad = -2.0 * lam * pos
        noise = rng.normal(0, noise_std, size=2)
        pos = pos + eta * (grad + noise)

        if coupled:
            s = rng.choice([-1, 1])
            force = gamma * s * (anchor - pos)
            pos += eta * force

        proj_bump[t + 1] = pos @ anchor_hat
        proj_orth[t + 1] = pos @ orth_hat

    return proj_bump, proj_orth


# ── Run ────────────────────────────────────────────────────────────────────
print("\nRunning uncoupled...")
unc_bump, unc_orth = run_2d(coupled=False, seed=42)
print("Running coupled...")
cpl_bump, cpl_orth = run_2d(coupled=True, seed=42)

# ── Measure variance ───────────────────────────────────────────────────────
n_burn = n_steps // 4
vars_measured = {
    'unc_bump': np.var(unc_bump[n_burn:]),
    'unc_orth': np.var(unc_orth[n_burn:]),
    'cpl_bump': np.var(cpl_bump[n_burn:]),
    'cpl_orth': np.var(cpl_orth[n_burn:]),
}
print(f"\nMeasured variance (last 75%):")
print(f"  Anchor dir: uncoupled={vars_measured['unc_bump']:.2f}  coupled={vars_measured['cpl_bump']:.2f}  ratio={vars_measured['cpl_bump']/vars_measured['unc_bump']:.1f}×")
print(f"  Orth dir:   uncoupled={vars_measured['unc_orth']:.2f}  coupled={vars_measured['cpl_orth']:.2f}  ratio={vars_measured['cpl_orth']/vars_measured['unc_orth']:.1f}×")

# ── 4-panel figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True)

steps = np.arange(n_steps + 1)
ylim = max(np.abs(cpl_bump).max(), np.abs(cpl_orth).max(),
           np.abs(unc_bump).max(), np.abs(unc_orth).max()) * 1.1
ylim = (-ylim, ylim)

panels = [
    (axes[0, 0], unc_bump, 'steelblue', 'Uncoupled — anchor direction'),
    (axes[0, 1], cpl_bump, 'steelblue', 'Coupled — anchor direction'),
    (axes[1, 0], unc_orth, 'forestgreen', 'Uncoupled — orthogonal direction'),
    (axes[1, 1], cpl_orth, 'forestgreen', 'Coupled — orthogonal direction'),
]

for ax, data, color, title in panels:
    ax.plot(steps, data, color=color, linewidth=0.3, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(ylim)

    # Variance annotation
    var_val = np.var(data[n_burn:])
    ax.text(0.02, 0.95, f'Var = {var_val:.1f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Anchor position marker on bump-direction panels
for ax in [axes[0, 0], axes[0, 1]]:
    ax.axhline(np.linalg.norm(anchor), color='red', linestyle='-',
               alpha=0.3, linewidth=1, label='Anchor')
    ax.legend(loc='upper right', fontsize=9)

axes[0, 0].set_ylabel('Position (anchor axis)', fontsize=11)
axes[1, 0].set_ylabel('Position (orthogonal axis)', fontsize=11)
axes[1, 0].set_xlabel('Step', fontsize=11)
axes[1, 1].set_xlabel('Step', fontsize=11)

plt.suptitle('Anisotropic Temperature from Martingale Coupling\n'
             f'Agent 0 pinned at ({anchor[0]:.0f}, {anchor[1]:.0f}); '
             f'coupling noise std = {eta*gamma*np.linalg.norm(anchor):.2f}, '
             f'SGD noise std = {eta*noise_std:.2f}',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_martingale_1d.pdf', bbox_inches='tight')
print("\nSaved fig_martingale_1d.pdf")
