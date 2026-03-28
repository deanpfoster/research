#!/usr/bin/env python3
"""Contour plot of a single rank-2 bump with SGD trace and edge-of-stability ring."""

import numpy as np
import matplotlib.pyplot as plt

b = 5.0
c = np.array([1.0, -0.5])
lam = 0.1
a = 4.0
A = a * np.eye(2)

def f(xy):
    d = xy - c
    q = 1 + d @ A @ d
    return b / q - lam * (xy @ xy)

def grad_f(xy):
    d = xy - c
    q = 1 + d @ A @ d
    return -b * 2 * A @ d / q**2 - 2 * lam * xy

def hessian_f(xy):
    """Hessian of f = b/q - lam|x|^2 where q = 1 + (x-c)'A(x-c)."""
    d = xy - c
    q = 1 + d @ A @ d
    grad_q = 2 * A @ d
    H = (b / q**2) * (2 / q * np.outer(grad_q, grad_q) - 2 * A) - 2 * lam * np.eye(2)
    return H

def max_curvature(xy):
    """Max eigenvalue of -Hessian (curvature for ascent)."""
    H = hessian_f(xy)
    return np.max(np.linalg.eigvalsh(-H))

# SGD: gradient ascent with noise, fixed learning rate
np.random.seed(7)
eta = 0.3
path = [np.array([-2.0, 2.5])]
for _ in range(200):
    g = grad_f(path[-1]) + np.random.normal(0, 0.3, size=2)
    path.append(path[-1] + eta * g)
path = np.array(path)

grid = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(grid, grid)
dx = X - c[0]
dy = Y - c[1]
quad = A[0,0]*dx**2 + (A[0,1]+A[1,0])*dx*dy + A[1,1]*dy**2
F = b / (1 + quad) - lam * (X**2 + Y**2)

# Compute max curvature on the grid for the edge-of-stability contour
curv = np.zeros_like(F)
for i in range(len(grid)):
    for j in range(len(grid)):
        curv[i, j] = max_curvature(np.array([grid[j], grid[i]]))

eos_threshold = 2 / eta

fig, ax = plt.subplots(figsize=(5, 4.5))
cs = ax.contourf(X, Y, F, levels=20, cmap='viridis')
ax.contour(X, Y, F, levels=20, colors='k', linewidths=0.3)
fig.colorbar(cs, ax=ax, label='$f(x)$')
# Edge of stability ring
ax.contour(X, Y, curv, levels=[eos_threshold], colors='orange',
           linewidths=2, linestyles='--')
ax.plot(path[:,0], path[:,1], 'r.-', markersize=3, linewidth=0.8, alpha=0.8)
ax.plot(path[0,0], path[0,1], 'go', markersize=7, zorder=5, label='start')
ax.plot(path[-1,0], path[-1,1], 'rs', markersize=7, zorder=5, label='end')
ax.plot(*c, 'w+', markersize=10, markeredgewidth=2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Rank-2 bump with SGD trajectory')
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='upper right')
fig.tight_layout()
fig.savefig('fig_rank2.pdf')
