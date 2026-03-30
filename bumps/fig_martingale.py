#!/usr/bin/env python3
"""
Martingale bump discovery: 100 agents in 2D, one starts in a bump.
Compare random search vs martingale-coupled search (IID +/-1 signs).

The landscape includes the global regularizer -lambda*||x||^2 from the
bump DGP, so agents have a restoring force and agent 0 stays near the
bump naturally (no artificial pinning).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── Bump landscape with regularizer ─────────────────────────────────────────
def f(xy, bumps, lam):
    """Evaluate bump landscape at xy (shape (N,2)) with regularizer."""
    val = -lam * np.sum(xy**2, axis=1)
    for b, c, A in bumps:
        d = xy - c[None, :]
        q = 1.0 + np.einsum('ij,jk,ik->i', d, A, d)
        val += b / q
    return val

def grad_f(xy, bumps, lam):
    """Gradient of bump landscape at xy (shape (N,2)) with regularizer."""
    g = -2.0 * lam * xy
    for b, c, A in bumps:
        d = xy - c[None, :]
        q = 1.0 + np.einsum('ij,jk,ik->i', d, A, d)
        g += -b * 2.0 * (d @ A) / (q**2)[:, None]
    return g

# ── Simulation parameters ──────────────────────────────────────────────────
n_agents = 100
n_steps = 5000
n_trials = 200
eta = 0.01          # learning rate for gradient ascent
noise_std = 0.5     # SGD noise (random walk scale)
gamma_per_agent = 0.001  # coupling per agent
lam = 0.0003        # global regularizer (weak enough that bump at (15,15) dominates)
domain = 40.0       # agents initialized in [-domain, domain]^2
discovery_radius = 3.0  # how close to bump center counts as "found"

# Bump at (15, 15) — regularizer penalty is lam*|c|^2 = 0.0003*450 = 0.135 << bump height
bump_center = np.array([15.0, 15.0])
bump_height = 5.0
bump_sharpness = 1.0  # sharper bump, tighter basin
A_bump = bump_sharpness * np.eye(2)
bumps = [(bump_height, bump_center, A_bump)]


def run_trial(mode, rng):
    """
    Run one trial. Returns the step at which a second agent discovers the bump.
    mode: 'random' or 'martingale'
    """
    pos = rng.uniform(-domain, domain, size=(n_agents, 2))
    pos[0] = bump_center + rng.normal(0, 0.3, size=2)

    if mode == 'martingale':
        # Fully IID coin flips
        sign_matrix = rng.choice([-1, 1], size=(n_agents, n_agents))
        np.fill_diagonal(sign_matrix, 0)

    for t in range(n_steps):
        # Check if any agent besides 0 is near the bump
        dists = np.linalg.norm(pos[1:] - bump_center[None, :], axis=1)
        if np.any(dists < discovery_radius):
            return t

        # Gradient step (ascent on the bump landscape)
        g = grad_f(pos, bumps, lam)
        noise = rng.normal(0, noise_std, size=pos.shape)
        pos = pos + eta * g + eta * noise

        # Coupling step (vectorized)
        if mode == 'martingale':
            diff = pos[None, :, :] - pos[:, None, :]  # (n, n, 2)
            forces = gamma_per_agent * np.einsum('ij,ijk->ik', sign_matrix, diff)
            pos += forces

    return n_steps  # not found


def run_trials(mode, n_trials, seed=42):
    """Run multiple trials and return discovery times."""
    rng = np.random.default_rng(seed)
    times = []
    for trial in range(n_trials):
        t = run_trial(mode, rng)
        times.append(t)
        if (trial + 1) % 50 == 0:
            print(f"  {mode}: {trial+1}/{n_trials} trials done")
    return np.array(times)


# ── Run experiments ─────────────────────────────────────────────────────────
print("Running random search trials...")
times_random = run_trials('random', n_trials, seed=42)
print("Running martingale-coupled trials...")
times_martingale = run_trials('martingale', n_trials, seed=123)

# ── Print summary statistics ────────────────────────────────────────────────
print("\n" + "="*60)
print("Discovery time (steps until 2nd agent finds the bump)")
print("="*60)
for label, times in [("Random search", times_random),
                     ("Martingale", times_martingale)]:
    found = times[times < n_steps]
    pct = 100 * len(found) / len(times)
    med = np.median(found) if len(found) > 0 else float('inf')
    mean = np.mean(found) if len(found) > 0 else float('inf')
    print(f"{label:20s}: median={med:6.0f}  mean={mean:6.0f}  "
          f"found={pct:5.1f}%  (of {len(times)} trials)")
print("="*60)

# ── Figure 1: Histogram comparison ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
bins = np.linspace(0, n_steps, 50)
ax.hist(times_random, bins=bins, alpha=0.6, label='Random search', color='steelblue')
ax.hist(times_martingale, bins=bins, alpha=0.6, label='Martingale coupling', color='coral')
ax.set_xlabel('Steps to discovery')
ax.set_ylabel('Count')
ax.set_title('Discovery time distribution')
ax.legend()

ax = axes[1]
for label, times, color in [("Random search", times_random, 'steelblue'),
                             ("Martingale", times_martingale, 'coral')]:
    sorted_t = np.sort(times)
    cdf = np.arange(1, len(sorted_t)+1) / len(sorted_t)
    ax.step(sorted_t, cdf, label=label, color=color, linewidth=2)
ax.set_xlabel('Steps')
ax.set_ylabel('Fraction discovered')
ax.set_title('Cumulative discovery probability')
ax.legend()

plt.tight_layout()
plt.savefig('fig_martingale_discovery.pdf', bbox_inches='tight')
print("Saved fig_martingale_discovery.pdf")

# ── Figure 2: Single trajectory snapshot ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, mode, title in [(axes[0], 'random', 'Random search (no coupling)'),
                         (axes[1], 'martingale', 'Martingale coupling')]:
    rng = np.random.default_rng(999)
    pos = rng.uniform(-domain, domain, size=(n_agents, 2))
    pos[0] = bump_center + rng.normal(0, 0.3, size=2)

    if mode == 'martingale':
        sign_matrix = rng.choice([-1, 1], size=(n_agents, n_agents))
        np.fill_diagonal(sign_matrix, 0)

    snapshot_step = 500
    for t in range(snapshot_step):
        g = grad_f(pos, bumps, lam)
        noise = rng.normal(0, noise_std, size=pos.shape)
        pos = pos + eta * g + eta * noise
        if mode == 'martingale':
            diff = pos[None, :, :] - pos[:, None, :]
            forces = gamma_per_agent * np.einsum('ij,ijk->ik', sign_matrix, diff)
            pos += forces

    grid = np.linspace(-domain, domain, 200)
    X, Y = np.meshgrid(grid, grid)
    xy_grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = f(xy_grid, bumps, lam).reshape(X.shape)
    ax.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.4)

    ax.scatter(pos[1:, 0], pos[1:, 1], s=15, c='steelblue', alpha=0.7, zorder=3)
    ax.scatter(pos[0, 0], pos[0, 1], s=80, c='red', marker='*', zorder=4,
               label='Agent 0 (in bump)')

    circle = Circle(bump_center, discovery_radius, fill=False,
                    edgecolor='red', linestyle='--', linewidth=1.5)
    ax.add_patch(circle)

    ax.set_xlim(-domain, domain)
    ax.set_ylim(-domain, domain)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\n(t = {snapshot_step})')
    ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('fig_martingale_snapshot.pdf', bbox_inches='tight')
print("Saved fig_martingale_snapshot.pdf")

# plt.show()  # non-interactive: only save to PDF
