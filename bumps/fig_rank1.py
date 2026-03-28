#!/usr/bin/env python3
"""Plot a single rank-1 bump with an SGD trace climbing toward the peak."""

import numpy as np
import matplotlib.pyplot as plt

b, c, a = 3.0, 1.0, 4.0

def f(x):
    return b / (1 + a * (x - c)**2 + x**2)

def grad_f(x):
    denom = 1 + a * (x - c)**2 + x**2
    return -b * (2 * a * (x - c) + 2 * x) / denom**2

# SGD: gradient ascent with noise to find the bump
np.random.seed(42)
eta = 0.3
x_sgd = [-3.0]
for _ in range(40):
    g = grad_f(x_sgd[-1]) + np.random.normal(0, 0.05)
    x_sgd.append(x_sgd[-1] + eta * g)
x_sgd = np.array(x_sgd)

x = np.linspace(-5, 5, 500)

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(x, f(x), 'k-', linewidth=1.5)
ax.plot(x_sgd, f(x_sgd), 'r.-', markersize=4, linewidth=0.8, alpha=0.7,
        label='SGD path')
ax.plot(x_sgd[0], f(x_sgd[0]), 'go', markersize=6, zorder=5, label='start')
ax.plot(x_sgd[-1], f(x_sgd[-1]), 'rs', markersize=6, zorder=5, label='end')
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_title('Rank-1 bump with SGD trajectory')
ax.axhline(0, color='gray', linewidth=0.5)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('fig_rank1.pdf')
