import numpy as np

def run_sim(tau, gamma):
    N = 100
    d = 10
    lmbda = 0.01
    eta = 0.01
    steps = 10000
    x = np.random.randn(N, d)
    
    for t in range(steps):
        if t % tau == 0:
            S = np.random.choice([-1, 1], size=(N, N))
            np.fill_diagonal(S, 0)
            
        g = 2 * lmbda * x
        noise = np.random.randn(N, d) * 0.1
        diff = x[None, :, :] - x[:, None, :]
        force = gamma * np.einsum('ij,ijk->ik', S, diff)
        
        x = x - eta * g + force + np.sqrt(2 * 0.01) * noise
        
        if np.max(np.linalg.norm(x, axis=1)) > 100:
            return f"EXPLODED at step {t}"
    
    return f"Stable. Max norm: {np.max(np.linalg.norm(x, axis=1)):.2f}"

print("tau=1, gamma=0.001:", run_sim(1, 0.001))
print("tau=50, gamma=0.001:", run_sim(50, 0.001))
print("tau=1, gamma=0.0001:", run_sim(1, 0.0001))
print("tau=50, gamma=0.0001:", run_sim(50, 0.0001))
