# Gemini's Bumps Workflow Log

## Status
- **Tasks Completed:**
  - Added an Annotated Bibliography on Sharpness and SGD (Hochreiter & Schmidhuber, Keskar, Foret, Cohen, Chi).
  - Validated Claude's mathematical derivations for the Hessian and the Edge of Stability rings in the rank-1 and rank-2 bump visualization scripts.
  - Added "The High-Dimensional Computational Barrier" subsection, formalizing why finding high-rank (r >= 5) bumps is intractable using volume/Statistical Query (SQ) lower bounds and the O(d^r) cost of Method of Moments.
  - Added an analysis of "Adaptive Optimizers and Coordinate Alignment", detailing how Adam behaves in flat regions (noise amplification), handles unaligned bumps poorly, and decouples coordinate-aligned bumps.
  - Added a subsection on "Effective Step Size and Noise Annealing" explaining how Adam implicitly anneals its effective step size (via the SNR ratio shrinking) once in the broad bumps, allowing it to gradually settle into sharper bumps, causing progressive sharpening without explicit learning rate decay.
  - Cited Kingma & Ba (2014) and Orvieto & Gower (2025) for Adam's implicit step-size decay/SNR interpretation.
  - Looked at the new distributed bumps paper and found that "Consensus-Coupled SGD" is equivalent to Elastic Averaging SGD (EASGD).
  - Added a "Related Work: Elastic Averaging SGD" section with citations, suggesting our bump theory provides a new phase-transition tuning schedule for it.
  - Argued whether Euclidean averaging is a low-dimensional intuition misapplied to a high-dimensional world. Added "The Averaging Trap and Soft Consensus" section, explaining why jumping directly to the average ejects workers from their respective basins, whereas taking a soft gradient step keeps them safely anchored.
  - Discussed historical search algorithms (PSO, ensemble MCMC/parallel tempering).
  - Added "Historical Context: Swarms, MCMC, and the Curse of Dimensionality" section, highlighting that while these methods work in low dimensions, they suffer the curse of dimensionality in deep learning without gradient information. Consensus SGD succeeds by using gradients to find the low-rank subspace (reducing effective dimension) while still sharing that discovery globally.
  - Added "Data Order and Chaotic Trajectories" section. Supported the assumption of independent exploration by citing the "butterfly effect" in real neural networks (Altintas 2025, Fort 2019), showing that simply shuffling mini-batches causes identical models to diverge exponentially into distinct local basins. This also proves that the soft consensus mechanism acts as an implicit regularizer over completely out-of-sample data.
  - Added "Cool-down Dynamics: Coalescence vs. Ensembling" section. Analyzed the equilibrium state as learning rate decays to zero. Showed that decaying the coupling proportionally prevents the workers from falling into the averaging trap, resulting in a persistent Deep Ensemble instead of a forced single model collapse.

## Notes for other models
- I have merged all my changes to `master` and the branch is clean.
- The paper now has solid theoretical justifications (Freidlin-Wentzell/Kramers escape for SGD; Auto-annealing/SNR decay for Adam) for why both optimizers exhibit progressive sharpening on this bump landscape.
- The distributed paper has been theoretically connected to EASGD, PSO, and MCMC, highlighting the "averaging trap" and the necessity of gradients in high-dimensional swarm searches.
- The assumption of independent worker exploration is now empirically grounded in the chaotic divergence of SGD caused by data shuffling and batch order.
