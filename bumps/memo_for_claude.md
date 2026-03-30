# Memo for Claude: Martingale-Coupled SGD

Hi Claude,

We've been iterating on the "Martingale-Coupled Distributed Search" paper and refining the sign matrix mechanism ($S_{ij}$). Here is the current state of our thinking based on recent discussions:

## The Change to Fully IID Coin Flips
We originally had a stratified/column-balanced approach where we forced exactly half the interactions for any given agent to be attractive ($+1$) and half to be repulsive ($-1$). 

We have **switched to fully IID coin flips** ($S_{ij}$ drawn uniformly from $\{-1, +1\}$). This removes the strict column balancing.

## The $\approx \log_2(n)$ Cascade and Recruiting Everyone
With fully independent IID coin flips, whenever we re-randomize the sign matrix (a new "phase"), roughly half the remaining exploring agents will randomly draw a net positive attraction toward the cluster in the bump and get pulled in. Thus, the population of explorers halves with each phase. Over $\approx \log_2(n)$ re-randomizations, the cascade will eventually recruit *every single agent* in the ensemble into the bump.

## Why a "Ghost Town" is Actually What We Want
In a standard low-dimensional optimization problem, recruiting everyone to a single minimum is catastrophic because the rest of the landscape becomes a "ghost town" and exploration stops. 

However, we are looking at a small piece of the overall puzzle. In our setting, the ambient dimension is vast ($d \sim 10^9$ to $10^{12}$), but the features/bumps are extremely low-rank (e.g., $r=2$). 

When the entire ensemble is recruited to a discovered bump, they are only collapsing in the $2$ dimensions that define that specific feature. In the remaining $10^{12} - 2$ dimensions (the vast null space of the bump), the agents remain completely unconstrained and independent! 

Therefore, we *want* to recruit everyone to the feature to exploit it fully; the massive orthogonal null space ensures that exploration for the next feature continues automatically.

## Updates Made
1. **`fig_martingale.py`**: Updated the sign matrix generation to use simple `rng.choice([-1, 1], size=(n_agents, n_agents))` instead of the loop that enforced column balancing.
2. **`martingale_bumps.tex`**: Updated the definition of $S$ and added paragraphs explaining the transition to fully IID flips and the low-rank / high-dimension justification for recruiting everyone.

I've committed these changes so you can review them on `master`. Let's coordinate on the next steps for the 2D Freidlin-Wentzell proofs!