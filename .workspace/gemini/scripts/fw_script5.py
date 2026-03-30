import sympy as sp

x, y, c, gamma, b, r = sp.symbols('x y c gamma b r', real=True)
L = -b / (1 + ((x-c)**2 + y**2)/r**2)
U = L + gamma/4 * (x**2 + y**2)

Hx = sp.diff(U, x, 2)
Hy = sp.diff(U, y, 2)

Hxx = Hx.subs(y, 0).simplify()
print("Hxx at y=0:", Hxx)
