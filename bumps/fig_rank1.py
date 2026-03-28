#!/usr/bin/env python3
"""Contour plot of a rank-1 bump in R^2 with SGD trajectory."""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(17)

b, c = 3.0, np.array([1.0, 0.0])
# Rank-1 shape matrix: only varies along direction v = (cos30, sin30)
v = np.array([np.cos(np.pi/6), np.sin(np.pi/6)])
a = 6.0
A = a * np.outer(v, v)  # rank 1

def f(xy):
    d = xy - c
    return b / (1 + d @ A @ d + xy @ xy)

def grad_f(xy):
    d = xy - c
    denom = 1 + d @ A @ d + xy @ xy
    return -b * (2 * A @ d + 2 * xy) / denom**2

# SGD gradient ascent with noise — start within reach of the gradient
eta = 1.5
path = [np.array([-2.0, 2.0])]
for _ in range(200):
    g = grad_f(path[-1]) + np.random.normal(0, 0.03, size=2)
    path.append(path[-1] + eta * g)
path = np.array(path)

# Contour plot
grid = np.linspace(-5, 5, 300)
X, Y = np.meshgrid(grid, grid)
dx = X - c[0]
dy = Y - c[1]
quad = A[0,0]*dx**2 + (A[0,1]+A[1,0])*dx*dy + A[1,1]*dy**2
norm2 = X**2 + Y**2
F = b / (1 + quad + norm2)

fig, ax = plt.subplots(figsize=(5, 4.5))
cs = ax.contourf(X, Y, F, levels=20, cmap='viridis')
ax.contour(X, Y, F, levels=20, colors='k', linewidths=0.3)
fig.colorbar(cs, ax=ax, label='$f(x)$')
ax.plot(path[:,0], path[:,1], 'r.-', markersize=3, linewidth=0.8, alpha=0.8)
ax.plot(path[0,0], path[0,1], 'go', markersize=7, zorder=5, label='start')
ax.plot(path[-1,0], path[-1,1], 'rs', markersize=7, zorder=5, label='end')
ax.plot(*c, 'w+', markersize=10, markeredgewidth=2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Rank-1 bump with SGD trajectory')
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='upper right')
fig.tight_layout()
fig.savefig('fig_rank1.pdf')
