import numpy as np
import matplotlib.pyplot as plt

def L(x, y):
    b1 = 1.0
    b2 = 1.0
    c1 = np.array([0, 0])
    c2 = np.array([5, 0])
    # Loss is negative of bump function
    bump1 = -b1 / (1 + (x - c1[0])**2 + (y - c1[1])**2)
    bump2 = -b2 / (1 + (x - c2[0])**2 + (y - c2[1])**2)
    return bump1 + bump2

def U_eff(x, y, gamma):
    return L(x, y) + (gamma / 4.0) * (x**2 + y**2)

x = np.linspace(-2, 7, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)
gamma = 0.1
Z = U_eff(X, Y, gamma)

plt.contourf(X, Y, Z, levels=50)
plt.colorbar()
plt.savefig('fw_test.png')
