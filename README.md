# Progressive Sharpening via Bump Functions

Research paper exploring how SGD progressively discovers features of increasing intrinsic rank in a bump-function landscape.

## Data Generating Process

f(x) = sum_i b_i / (1 + (x - c_i)' A_i (x - c_i) + ||x||^2)

where the shape matrices A_i are rank-deficient, controlling the intrinsic dimensionality of each bump. Rank-1 bumps are ridges (easy to find), higher-rank bumps have smaller basins of attraction (harder to find), and SGD naturally discovers them in order of increasing rank --- progressive sharpening.

## Building

```
cd bumps
make        # generates figures from Python, then builds bumps.pdf
make clean  # removes all generated files
```

Requires: Python 3 with numpy/matplotlib, pdflatex, bibtex.

## Workflow

Multiple AI collaborators (Claude, Gemini, GPT) work on branches and merge to master via fast-forward. See `.workspace/WORKFLOW.md` for the protocol.
