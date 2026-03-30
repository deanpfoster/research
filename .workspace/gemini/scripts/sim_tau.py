import numpy as np
import matplotlib.pyplot as plt

def L_grad(x, lmbda):
    return 2 * lmbda * x

N = 100
d = 10
lmbda = 0.01
eta = 0.01
gamma = 0.05
steps = 50000

# Try tau=1 and tau=50
for tau in [1, 50]:
    x = np.random.randn(N, d)
    max_norm = []
    
    for t in range(steps):
        if t % tau == 0:
            S = np.random.choice([-1, 1], size=(N, N))
            np.fill_diagonal(S, 0)
            
        g = L_grad(x, lmbda)
        noise = np.random.randn(N, d) * 0.1
        
        diff = x[None, :, :] - x[:, None, :]
        force = gamma * np.einsum('ij,ijk->ik', S, diff)
        
        x = x - eta * g + force + eta * noise
        max_norm.append(np.max(np.linalg.norm(x, axis=1)))
        
        if np.max(np.linalg.norm(x, axis=1)) > 1000:
            print(f"tau={tau} EXPLODED at step {t}")
            break
            
    print(f"tau={tau} max norm reached: {max_norm[-1]:.2f}")
