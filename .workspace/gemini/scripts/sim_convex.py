import numpy as np

def run_convex_sim(n, gamma):
    d = 10
    eta = 0.01
    H = np.diag(np.linspace(0.1, 1.0, d))
    steps = 20000
    
    x = np.random.randn(n, d) * 10.0
    
    for t in range(steps):
        S = np.random.choice([-1, 1], size=(n, n))
        np.fill_diagonal(S, 0)
        
        # True gradient
        g = x @ H
        # SGD mini-batch noise
        noise = np.random.randn(n, d) * 1.0
        
        diff = x[None, :, :] - x[:, None, :]
        force = gamma * np.einsum('ij,ijk->ik', S, diff)
        
        x = x - eta * g + force + eta * noise
        
    # Measure steady-state loss (1/2 x^T H x)
    loss = 0.5 * np.mean(np.sum(x * (x @ H), axis=1))
    return loss

print("Uncoupled (gamma=0):", run_convex_sim(20, 0.0))
print("Coupled (gamma=0.01):", run_convex_sim(20, 0.01))
print("Coupled (gamma=0.03):", run_convex_sim(20, 0.03))
