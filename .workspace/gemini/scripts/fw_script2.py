import sympy as sp

x, y, gamma, b, r, c = sp.symbols('x y gamma b r c', real=True)
# Let's do 1D for simplicity first to see the saddle point
L = -b / (1 + (x-c)**2 / r**2)
U = L + gamma/4 * x**2

dU = sp.diff(U, x)
print("dU/dx =", dU)
