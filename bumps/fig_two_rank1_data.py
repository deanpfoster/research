#!/usr/bin/env python3
"""Data-distributed view of the two rank-1 X-shaped bumps.
Shows what a learner actually sees: scatter of sampled points colored by
f(x)+noise, with increasing sample sizes so the X structure progressively
emerges."""

import numpy as np
import matplotlib.pyplot as plt

# Same DGP as fig_two_rank1.py
center = np.array([0.5, 0.0])
b1, b2 = 5.0, 3.0

angle1, angle2 = np.pi / 4, -np.pi / 4
v1 = np.array([np.cos(angle1), np.sin(angle1)])
v2 = np.array([np.cos(angle2), np.sin(angle2)])
a1, a2 = 6.0, 6.0
A1 = a1 * np.outer(v1, v1)
A2 = a2 * np.outer(v2, v2)

def f(xy):
    d = xy - center
    g1 = 1 + d @ A1 @ d + xy @ xy
    g2 = 1 + d @ A2 @ d + xy @ xy
    return b1 / g1 + b2 / g2

# Sample sizes showing progressive resolution of the X
sample_sizes = [50, 200, 1000, 5000]
noise_std = 0.10
xlim = 3.0  # tighter window so more samples hit the ridge

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2),
                         gridspec_kw={'wspace': 0.05, 'right': 0.92})
rng = np.random.default_rng(42)

point_sizes = {50: 30, 200: 18, 1000: 8, 5000: 3}

for ax, N in zip(axes, sample_sizes):
    # Sample uniformly in [-xlim, xlim]^2
    xs = rng.uniform(-xlim, xlim, size=(N, 2))
    ys = np.array([f(x) for x in xs]) + rng.normal(0, noise_std, size=N)

    sc = ax.scatter(xs[:, 0], xs[:, 1], c=ys, s=point_sizes[N],
                    cmap='viridis', alpha=0.8, edgecolors='none',
                    vmin=0, vmax=5.0)
    ax.set_aspect('equal')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-xlim, xlim)
    ax.set_title(f'$N = {N}$', fontsize=11)
    ax.set_xlabel('$x_1$', fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel('$x_2$', fontsize=9)
    else:
        ax.set_yticklabels([])

cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
fig.colorbar(sc, cax=cbar_ax, label='$y = f(x) + \\varepsilon$')
fig.suptitle('Data-distributed view: X-shaped ridge emerges with sample size',
             fontsize=13)
fig.savefig('fig_two_rank1_data.pdf', bbox_inches='tight')
