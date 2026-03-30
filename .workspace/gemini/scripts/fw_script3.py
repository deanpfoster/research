import sympy as sp
sp.init_printing()

# Let's derive the exact joint potential and its FW action
X1_x, X1_y, X2_x, X2_y, gamma, D = sp.symbols('X1_x X1_y X2_x X2_y gamma D')

# Joint potential
def U(x1, y1, x2, y2):
    return (gamma/4) * ((x1 - x2)**2 + (y1 - y2)**2)

# If X1 is at 0,0
U_eff = U(0, 0, X2_x, X2_y)
print("U_eff =", U_eff)
