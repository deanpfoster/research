#!/usr/bin/env python3
"""Plot a single rank-1 bump: f(x) = b / (1 + a*(x-c)^2 + x^2)."""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)
b, c, a = 3.0, 1.0, 4.0
f = b / (1 + a * (x - c)**2 + x**2)

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(x, f, 'k-', linewidth=1.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_title('Rank-1 bump: $f(x) = b / (1 + a(x-c)^2 + x^2)$')
ax.axhline(0, color='gray', linewidth=0.5)
fig.tight_layout()
fig.savefig('fig_rank1.pdf')
