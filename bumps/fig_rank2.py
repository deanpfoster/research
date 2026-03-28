#!/usr/bin/env python3
"""Contour plot of a single rank-2 bump with an SGD trace."""

import numpy as np
import matplotlib.pyplot as plt

b = 5.0
c = np.array([1.0, -0.5])
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
D = np.diag([6.0, 1.0])
A = R @ D @ R.T

def f(xy):
    d = xy - c
    return b / (1 + d @ A @ d + xy @ xy)

def grad_f(xy):
    d = xy - c
    denom = 1 + d @ A @ d + xy @ xy
    return -b * (2 * A @ d + 2 * xy) / denom**2

# SGD: gradient ascent with noise and mild decay
np.random.seed(7)
eta0 = 0.8
path = [np.array([-2.0, 2.5])]
for t in range(200):
    eta = eta0 / (1 + t / 100)
    g = grad_f(path[-1]) + np.random.normal(0, 0.03, size=2)
    path.append(path[-1] + eta * g)
path = np.array(path)

grid = np.linspace(-4, 4, 300)
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
ax.set_title('Rank-2 bump with SGD trajectory')
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='upper right')
fig.tight_layout()
fig.savefig('fig_rank2.pdf')
