#!/usr/bin/env python3
"""3x3 grid: two rank-1 bumps forming an X, SGD finds one ridge then the center.
Path colored by time: blue (early) -> magenta (late)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# Two rank-1 components sharing a center, ridges at different angles
# Ridge 1: broad and tall -> found first
# Ridge 2: sharper but shorter -> found second at the crossing
center = np.array([0.5, 0.0])
b1, b2 = 5.0, 3.0  # taller ridge found first
lam = 0.1  # global regularizer

angle1, angle2 = np.pi / 4, -np.pi / 4
v1 = np.array([np.cos(angle1), np.sin(angle1)])
v2 = np.array([np.cos(angle2), np.sin(angle2)])
a1, a2 = 6.0, 6.0  # both sharp so X is visible
A1 = a1 * np.outer(v1, v1)
A2 = a2 * np.outer(v2, v2)

def f(xy):
    d = xy - center
    q1 = 1 + d @ A1 @ d
    q2 = 1 + d @ A2 @ d
    return b1 / q1 + b2 / q2 - lam * (xy @ xy)

def grad_f(xy):
    d = xy - center
    q1 = 1 + d @ A1 @ d
    q2 = 1 + d @ A2 @ d
    return (-b1 * 2 * A1 @ d / q1**2
            - b2 * 2 * A2 @ d / q2**2
            - 2 * lam * xy)

def hessian_f(xy):
    d = xy - center
    H = -2 * lam * np.eye(2)
    for Ai, bi in [(A1, b1), (A2, b2)]:
        q = 1 + d @ Ai @ d
        grad_q = 2 * Ai @ d
        H += (bi / q**2) * (2 / q * np.outer(grad_q, grad_q) - 2 * Ai)
    return H

def max_curvature(xy):
    return np.max(np.linalg.eigvalsh(-hessian_f(xy)))

# Precompute surface and curvature
grid = np.linspace(-5, 5, 150)
X, Y = np.meshgrid(grid, grid)
F = np.zeros_like(X)
curv = np.zeros_like(X)
for i in range(len(grid)):
    for j in range(len(grid)):
        xy = np.array([grid[j], grid[i]])
        F[i, j] = f(xy)
        curv[i, j] = max_curvature(xy)

eta = 0.15
eos = 2 / eta
nsteps = 800

# 9 starting points — mostly along the broad ridge's axis so they
# absorb into it naturally before sliding to the center
starts = [
    np.array([-3.5,  3.5]),
    np.array([ 3.5,  3.5]),
    np.array([-4.0,  0.0]),
    np.array([ 4.0,  0.0]),
    np.array([-3.5, -3.5]),
    np.array([ 3.5, -3.5]),
    np.array([ 0.0,  4.0]),
    np.array([ 0.0, -4.0]),
    np.array([-2.0,  3.0]),
]

fig, axes = plt.subplots(3, 3, figsize=(12, 11))
cmap = plt.cm.cool

for idx, start in enumerate(starts):
    ax = axes[idx // 3, idx % 3]

    np.random.seed(200 + idx)
    path = [start.copy()]
    for _ in range(nsteps):
        g = grad_f(path[-1]) + np.random.normal(0, 0.3, size=2)
        path.append(path[-1] + eta * g)
    path = np.array(path)

    ax.contourf(X, Y, F, levels=20, cmap='viridis')
    ax.contour(X, Y, F, levels=20, colors='k', linewidths=0.2)
    ax.contour(X, Y, curv, levels=[eos], colors='orange',
               linewidths=1.5, linestyles='--')

    # Color path by time
    segments = np.stack([path[:-1], path[1:]], axis=1)
    norm = Normalize(0, nsteps)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=0.7, alpha=0.8)
    lc.set_array(np.arange(nsteps))
    ax.add_collection(lc)

    ax.plot(path[0,0], path[0,1], 'go', markersize=5, zorder=5)
    ax.plot(path[-1,0], path[-1,1], 'rs', markersize=5, zorder=5)
    ax.plot(*center, 'w+', markersize=8, markeredgewidth=1.5)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(f'seed {200+idx}', fontsize=9)
    if idx // 3 == 2:
        ax.set_xlabel('$x_1$', fontsize=8)
    if idx % 3 == 0:
        ax.set_ylabel('$x_2$', fontsize=8)

fig.suptitle('Two rank-1 bumps ($a_1{=}a_2{=}6$, $b_1{=}5$, $b_2{=}3$): blue (early) $\\to$ magenta (late)',
             fontsize=12)
fig.tight_layout()
fig.savefig('fig_two_rank1.pdf')
