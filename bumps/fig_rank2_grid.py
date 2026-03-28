#!/usr/bin/env python3
"""3x3 grid of rank-2 bumps with different shapes, SGD paths, and EoS rings."""

import numpy as np
import matplotlib.pyplot as plt

lam = 0.1

def make_bump(angle, center, height, eig1, eig2):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    A = R @ np.diag([eig1, eig2]) @ R.T
    return A, center, height

def f(xy, A, c, b):
    d = xy - c
    return b / (1 + d @ A @ d) - lam * (xy @ xy)

def grad_f(xy, A, c, b):
    d = xy - c
    q = 1 + d @ A @ d
    return -b * 2 * A @ d / q**2 - 2 * lam * xy

def max_curvature(xy, A, c, b):
    d = xy - c
    q = 1 + d @ A @ d
    grad_q = 2 * A @ d
    H = (b / q**2) * (2 / q * np.outer(grad_q, grad_q) - 2 * A) - 2 * lam * np.eye(2)
    return np.max(np.linalg.eigvalsh(-H))

# 9 configurations: vary orientation, eccentricity, center
configs = [
    (0,          np.array([ 0.5,  0.0]), 5.0, 4.0, 4.0),   # round
    (np.pi/6,    np.array([ 0.8,  0.3]), 5.0, 6.0, 2.0),   # mild ellipse
    (np.pi/3,    np.array([ 0.0,  0.5]), 5.0, 8.0, 3.0),   # steeper
    (np.pi/2,    np.array([ 0.3, -0.2]), 4.0, 4.0, 4.0),   # round, shifted
    (2*np.pi/3,  np.array([-0.3,  0.4]), 5.0, 5.0, 2.0),   # ellipse
    (5*np.pi/6,  np.array([ 0.0,  0.0]), 6.0, 3.0, 3.0),   # round, tall
    (np.pi/4,    np.array([ 0.6, -0.5]), 4.0, 7.0, 1.5),   # very eccentric
    (3*np.pi/8,  np.array([-0.4,  0.2]), 5.0, 5.0, 5.0),   # round
    (7*np.pi/12, np.array([ 0.2,  0.6]), 5.0, 6.0, 3.0),   # ellipse
]

grid = np.linspace(-4, 4, 150)
X, Y = np.meshgrid(grid, grid)
eta = 0.3

fig, axes = plt.subplots(3, 3, figsize=(12, 11))

for idx, (angle, center, height, eig1, eig2) in enumerate(configs):
    ax = axes[idx // 3, idx % 3]
    A, c, b = make_bump(angle, center, height, eig1, eig2)

    # Surface
    F = np.zeros_like(X)
    curv = np.zeros_like(X)
    for i in range(len(grid)):
        for j in range(len(grid)):
            xy = np.array([grid[j], grid[i]])
            F[i, j] = f(xy, A, c, b)
            curv[i, j] = max_curvature(xy, A, c, b)

    # SGD with decay
    np.random.seed(7 + idx)
    start = np.array([-2.5, 2.5]) if idx % 2 == 0 else np.array([2.5, -2.0])
    path = [start.copy()]
    for _ in range(200):
        g = grad_f(path[-1], A, c, b) + np.random.normal(0, 0.3, size=2)
        path.append(path[-1] + eta * g)
    path = np.array(path)

    eos = 2 / eta

    ax.contourf(X, Y, F, levels=15, cmap='viridis')
    ax.contour(X, Y, F, levels=15, colors='k', linewidths=0.2)
    ax.contour(X, Y, curv, levels=[eos], colors='orange',
               linewidths=1.5, linestyles='--')
    ax.plot(path[:,0], path[:,1], 'r.-', markersize=1.5, linewidth=0.5, alpha=0.7)
    ax.plot(path[0,0], path[0,1], 'go', markersize=4, zorder=5)
    ax.plot(path[-1,0], path[-1,1], 'rs', markersize=4, zorder=5)
    ax.plot(*c, 'w+', markersize=8, markeredgewidth=1.5)
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    deg = int(np.degrees(angle))
    ratio = f'{eig1:.0f}:{eig2:.0f}'
    ax.set_title(f'$\\theta={deg}°$, $\\lambda={ratio}$', fontsize=9)
    if idx // 3 == 2:
        ax.set_xlabel('$x_1$', fontsize=8)
    if idx % 3 == 0:
        ax.set_ylabel('$x_2$', fontsize=8)

fig.suptitle('Rank-2 bumps: varying orientation and eigenvalue ratio', fontsize=13)
fig.tight_layout()
fig.savefig('fig_rank2_grid.pdf')
