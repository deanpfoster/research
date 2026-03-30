#!/usr/bin/env python3
"""
Martingale recruitment cascade with fitness-weighted coupling.

All-pairs coupling with IID +/-1 signs, but each agent's broadcast is
weighted by its own function value. Agents at bumps (high f) exert strong
coupling; agents in flat regions (f ≈ 0) exert almost none. This creates
a natural signal-to-noise filter: the bump dominates the coupling without
anyone needing to know which agents are "discoverers."

Sign matrix is re-randomized periodically to maintain the martingale.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── Bump landscape with regularizer ─────────────────────────────────────────
def f_vec(xy, bumps, lam):
    """Evaluate at array of points (N,2)."""
    val = -lam * np.sum(xy**2, axis=1)
    for b, c, A in bumps:
        d = xy - c[None, :]
        q = 1.0 + np.einsum('ij,jk,ik->i', d, A, d)
        val += b / q
    return val

def grad_f(xy, bumps, lam):
    g = -2.0 * lam * xy
    for b, c, A in bumps:
        d = xy - c[None, :]
        q = 1.0 + np.einsum('ij,jk,ik->i', d, A, d)
        g += -b * 2.0 * (d @ A) / (q**2)[:, None]
    return g

# ── Parameters ──────────────────────────────────────────────────────────────
n_agents = 100
eta = 0.05
noise_std = 0.3
gamma = 0.003           # coupling strength (fitness-weighted)
lam = 0.0001            # weak regularizer
domain = 40.0
recruit_radius = 5.0
phase_length = 100      # re-randomize signs every this many steps

bump_center = np.array([15.0, 15.0])
bump_height = 8.0
A_bump = 1.5 * np.eye(2)
bumps = [(bump_height, bump_center, A_bump)]

print(f"At bump peak: f = {bump_height - lam * np.sum(bump_center**2):.2f}")
print(f"At origin:    f = {f_vec(np.zeros((1,2)), bumps, lam)[0]:.2f}")
print(f"At (-20,-20): f = {f_vec(np.array([[-20,-20]]), bumps, lam)[0]:.4f}")


def run_and_record(mode, n_steps, rng):
    """Run simulation with fitness-weighted martingale coupling."""
    pos = rng.uniform(-domain, domain, size=(n_agents, 2))
    pos[0] = bump_center + rng.normal(0, 0.1, size=2)

    if mode == 'martingale':
        sign_matrix = rng.choice([-1, 1], size=(n_agents, n_agents)).astype(float)
        np.fill_diagonal(sign_matrix, 0)

    snapshots = []
    recruited_counts = []
    snapshot_times = {0, 50, 150, 300, 600, 1000, 1500, 2500, 4000, n_steps}

    for t in range(n_steps + 1):
        dists = np.linalg.norm(pos - bump_center[None, :], axis=1)
        recruited_counts.append(np.sum(dists < recruit_radius))

        if t in snapshot_times:
            snapshots.append((t, pos.copy(), dists.copy()))

        if t == n_steps:
            break

        # Re-randomize sign matrix periodically
        if mode == 'martingale' and t > 0 and t % phase_length == 0:
            sign_matrix = rng.choice([-1, 1], size=(n_agents, n_agents)).astype(float)
            np.fill_diagonal(sign_matrix, 0)

        # Gradient ascent + noise
        g = grad_f(pos, bumps, lam)
        noise = rng.normal(0, noise_std, size=pos.shape)
        pos = pos + eta * (g + noise)

        # Fitness-weighted all-pairs coupling
        if mode == 'martingale':
            fitness = f_vec(pos, bumps, lam)
            weights = np.maximum(fitness, 0)  # only positive-fitness agents broadcast

            # force on i = gamma * sum_j w_j * S[i,j] * (pos[j] - pos[i])
            diff = pos[None, :, :] - pos[:, None, :]  # (n, n, 2): diff[i,j] = pos[j]-pos[i]
            weighted_signs = sign_matrix * weights[None, :]  # (n, n): weight by broadcaster j
            forces = gamma * np.einsum('ij,ijk->ik', weighted_signs, diff)
            pos += eta * forces

        # Pin agent 0 at bump
        pos[0] = bump_center + rng.normal(0, 0.3, size=2)

    return snapshots, np.array(recruited_counts)


# ── Run trials ──────────────────────────────────────────────────────────────
n_steps = 5000
n_trials = 100

print(f"\nRunning {n_trials} trials with n={n_agents} agents...")
all_counts_mart = []
all_counts_rand = []
for trial in range(n_trials):
    _, counts_m = run_and_record('martingale', n_steps, np.random.default_rng(1000 + trial))
    _, counts_r = run_and_record('random', n_steps, np.random.default_rng(1000 + trial))
    all_counts_mart.append(counts_m)
    all_counts_rand.append(counts_r)
    if (trial + 1) % 25 == 0:
        print(f"  {trial+1}/{n_trials} trials done")

avg_mart = np.mean(all_counts_mart, axis=0)
avg_rand = np.mean(all_counts_rand, axis=0)
p25_mart = np.percentile(all_counts_mart, 25, axis=0)
p75_mart = np.percentile(all_counts_mart, 75, axis=0)
p25_rand = np.percentile(all_counts_rand, 25, axis=0)
p75_rand = np.percentile(all_counts_rand, 75, axis=0)

# Single detailed run
print("Running detailed single trial...")
snaps_mart, counts_single_mart = run_and_record(
    'martingale', n_steps, np.random.default_rng(7))
snaps_rand, counts_single_rand = run_and_record(
    'random', n_steps, np.random.default_rng(7))

# Precompute landscape
grid = np.linspace(-domain, domain, 150)
X, Y = np.meshgrid(grid, grid)
xy_grid = np.stack([X.ravel(), Y.ravel()], axis=1)
Z = f_vec(xy_grid, bumps, lam).reshape(X.shape)

# ── Figure 1: Cascade snapshots ─────────────────────────────────────────────
n_panels = min(8, len(snaps_mart))
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes_flat = axes.ravel()

for idx in range(n_panels):
    ax = axes_flat[idx]
    t, pos, dists = snaps_mart[idx]

    ax.contourf(X, Y, Z, levels=15, cmap='YlOrRd', alpha=0.3)

    near = dists < recruit_radius
    far = ~near
    n_near = int(np.sum(near))

    if np.any(far):
        ax.scatter(pos[far, 0], pos[far, 1], s=15, c='steelblue',
                   alpha=0.6, zorder=3, label=f'Exploring ({int(np.sum(far))})')
    if np.any(near):
        ax.scatter(pos[near, 0], pos[near, 1], s=30, c='crimson',
                   alpha=0.9, zorder=4, label=f'At bump ({n_near})')

    circle = Circle(bump_center, recruit_radius, fill=False,
                    edgecolor='red', linestyle='--', linewidth=1.5, zorder=5)
    ax.add_patch(circle)

    ax.set_xlim(-domain, domain)
    ax.set_ylim(-domain, domain)
    ax.set_aspect('equal')
    ax.set_title(f't = {t}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.8)

fig.suptitle(f'Fitness-Weighted Martingale Recruitment (n={n_agents})',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('fig_recruitment_cascade.pdf', bbox_inches='tight')
print("Saved fig_recruitment_cascade.pdf")

# ── Figure 2: Averaged recruitment curves ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

steps = np.arange(len(avg_mart))
ax.plot(steps, avg_mart, color='coral', linewidth=2.5,
        label='Fitness-weighted martingale')
ax.fill_between(steps, p25_mart, p75_mart, color='coral', alpha=0.15)
ax.plot(steps, avg_rand, color='steelblue', linewidth=2.5,
        label='Random search (no coupling)')
ax.fill_between(steps, p25_rand, p75_rand, color='steelblue', alpha=0.15)

ax.axhline(n_agents, color='gray', linestyle=':', alpha=0.5)
ax.text(n_steps * 0.85, n_agents + 2, f'n = {n_agents}', color='gray', fontsize=10)

ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Agents near bump', fontsize=12)
ax.set_title(f'Average Recruitment Over {n_trials} Trials', fontsize=13)
ax.legend(fontsize=12)
ax.set_ylim(0, n_agents + 10)
plt.tight_layout()
plt.savefig('fig_recruitment_curve.pdf', bbox_inches='tight')
print("Saved fig_recruitment_curve.pdf")

# ── Figure 3: Side-by-side comparison ───────────────────────────────────────
compare_times = [0, 600, 1500, 4000]
fig, axes = plt.subplots(2, len(compare_times), figsize=(5 * len(compare_times), 10))

for col, target_t in enumerate(compare_times):
    for row, (label, snaps) in enumerate([('Random search', snaps_rand),
                                           ('Martingale', snaps_mart)]):
        ax = axes[row, col]
        best = min(snaps, key=lambda s: abs(s[0] - target_t))
        t, pos, dists = best[:3]

        ax.contourf(X, Y, Z, levels=15, cmap='YlOrRd', alpha=0.3)

        near = dists < recruit_radius
        far = ~near
        if np.any(far):
            ax.scatter(pos[far, 0], pos[far, 1], s=15, c='steelblue',
                       alpha=0.6, zorder=3)
        if np.any(near):
            ax.scatter(pos[near, 0], pos[near, 1], s=30, c='crimson',
                       alpha=0.9, zorder=4)

        circle = Circle(bump_center, recruit_radius, fill=False,
                        edgecolor='red', linestyle='--', linewidth=1.5, zorder=5)
        ax.add_patch(circle)

        ax.set_xlim(-domain, domain)
        ax.set_ylim(-domain, domain)
        ax.set_aspect('equal')

        if row == 0:
            ax.set_title(f't = {t}', fontsize=13, fontweight='bold')
        if col == 0:
            ax.set_ylabel(label, fontsize=13, fontweight='bold')

        n_recruited = int(np.sum(near))
        ax.text(0.02, 0.98, f'{n_recruited} at bump',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                va='top', color='crimson',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.suptitle('Random Search vs Fitness-Weighted Martingale',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_recruitment_comparison.pdf', bbox_inches='tight')
print("Saved fig_recruitment_comparison.pdf")

# plt.show()
