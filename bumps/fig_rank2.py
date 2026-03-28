#!/usr/bin/env python3
"""Contour plot of a single rank-2 bump in R^2."""

import numpy as np
import matplotlib.pyplot as plt

grid = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(grid, grid)

b = 5.0
c = np.array([1.0, -0.5])
# Shape matrix: elongated ellipse rotated 30 degrees
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
D = np.diag([6.0, 1.0])
A = R @ D @ R.T

dx = X - c[0]
dy = Y - c[1]
quad = A[0,0]*dx**2 + (A[0,1]+A[1,0])*dx*dy + A[1,1]*dy**2
norm2 = X**2 + Y**2
F = b / (1 + quad + norm2)

fig, ax = plt.subplots(figsize=(5, 4.5))
cs = ax.contourf(X, Y, F, levels=20, cmap='viridis')
ax.contour(X, Y, F, levels=20, colors='k', linewidths=0.3)
fig.colorbar(cs, ax=ax, label='$f(x)$')
ax.plot(*c, 'r+', markersize=10, markeredgewidth=2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Rank-2 bump (contour plot)')
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig('fig_rank2.pdf')
