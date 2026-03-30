#!/usr/bin/env python3
"""
Companion to fig_martingale_1d.py: same parameters and noise, but now
with a bump at the anchor point. Shows the coupled agent getting captured
while the uncoupled agent stays trapped near the origin.

Overlays the no-bump trace (faint) to show trajectories are identical
until the bump's basin of attraction grabs the coupled agent.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Parameters (identical to fig_martingale_1d.py) ─────────────────────────
eta = 0.1
noise_std = 1.0
gamma = 0.3
lam = 0.005

anchor = np.array([10.0, 0.0])
anchor_hat = anchor / np.linalg.norm(anchor)
orth_hat = np.array([0.0, 1.0])

# Bump at the anchor point
bump_height = 5.0
A_bump = 0.3 * np.eye(2)
bumps = [(bump_height, anchor, A_bump)]

n_steps = 20000


def grad_f(xy, bumps_list, lam_val):
    """Gradient of f at a single 2D point (ascent direction)."""
    g = -2.0 * lam_val * xy
    for b, c, A in bumps_list:
        d = xy - c
        q = 1.0 + d @ A @ d
        g += -b * 2.0 * (A @ d) / q**2
    return g


def run_2d(coupled, use_bump, sgd_noise, signs):
    """Run agent with pre-generated noise so streams are identical."""
    pos = np.zeros(2)
    active_bumps = bumps if use_bump else []

    proj_bump = np.zeros(n_steps + 1)
    proj_orth = np.zeros(n_steps + 1)

    for t in range(n_steps):
        g = grad_f(pos, active_bumps, lam)
        pos = pos + eta * (g + sgd_noise[t])

        if coupled:
            force = gamma * signs[t] * (anchor - pos)
            pos += eta * force

        proj_bump[t + 1] = pos @ anchor_hat
        proj_orth[t + 1] = pos @ orth_hat

    return proj_bump, proj_orth


# ── Pre-generate all noise (shared across conditions) ─────────────────────
print("Running 4 conditions (identical noise draws)...")
rng = np.random.default_rng(42)
sgd_noise = rng.normal(0, noise_std, size=(n_steps, 2))
signs = rng.choice([-1, 1], size=n_steps)

unc_nobump_b, unc_nobump_o = run_2d(False, False, sgd_noise, signs)
unc_bump_b,   unc_bump_o   = run_2d(False, True,  sgd_noise, signs)
cpl_nobump_b, cpl_nobump_o = run_2d(True,  False, sgd_noise, signs)
cpl_bump_b,   cpl_bump_o   = run_2d(True,  True,  sgd_noise, signs)

# ── Figure: 2 panels, bump direction only ──────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, sharey=True)
steps = np.arange(n_steps + 1)
bump_pos = np.linalg.norm(anchor)

# Top: uncoupled
ax = axes[0]
ax.plot(steps, unc_nobump_b, color='steelblue', linewidth=0.3, alpha=0.3,
        label='No bump (temperature only)')
ax.plot(steps, unc_bump_b, color='steelblue', linewidth=0.3, alpha=0.8,
        label='With bump')
ax.axhline(bump_pos, color='red', linestyle='-', alpha=0.3, linewidth=1,
           label='Bump center')
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.set_ylabel('Position (bump axis)', fontsize=11)
ax.set_title('Uncoupled — bump direction', fontsize=12)
ax.set_ylim(-20, 20)
ax.legend(loc='upper right', fontsize=9)

# Bottom: coupled
ax = axes[1]
ax.plot(steps, cpl_nobump_b, color='steelblue', linewidth=0.3, alpha=0.3,
        label='No bump (temperature only)')
ax.plot(steps, cpl_bump_b, color='steelblue', linewidth=0.3, alpha=0.8,
        label='With bump')
ax.axhline(bump_pos, color='red', linestyle='-', alpha=0.3, linewidth=1,
           label='Bump center')
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.set_ylabel('Position (bump axis)', fontsize=11)
ax.set_xlabel('Step', fontsize=11)
ax.set_title('Coupled — bump direction', fontsize=12)
ax.legend(loc='upper right', fontsize=9)

# Mark capture time for coupled
near = np.abs(cpl_bump_b - bump_pos) < 2.0
if np.any(near):
    t_hit = np.argmax(near)
    ax.axvline(t_hit, color='coral', linestyle=':', alpha=0.5)
    ax.text(t_hit + 200, -17, f'Captured: t={t_hit}', fontsize=10, color='coral')

# Mark if uncoupled ever finds it
near_unc = np.abs(unc_bump_b - bump_pos) < 2.0
if np.any(near_unc):
    t_hit_unc = np.argmax(near_unc)
    axes[0].axvline(t_hit_unc, color='coral', linestyle=':', alpha=0.5)
    axes[0].text(t_hit_unc + 200, -17, f'Captured: t={t_hit_unc}',
                 fontsize=10, color='coral')

plt.suptitle('Same noise, with and without bump\n'
             'Faint trace = no bump (from temperature figure); '
             'solid trace = bump adds capture',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_martingale_capture.pdf', bbox_inches='tight')
print("\nSaved fig_martingale_capture.pdf")

# Summary
print(f"\nCoupled with bump: ", end="")
if np.any(near):
    print(f"captured at t={np.argmax(near)}")
else:
    print("never captured")

print(f"Uncoupled with bump: ", end="")
if np.any(near_unc):
    print(f"captured at t={np.argmax(near_unc)}")
else:
    print("never captured")
