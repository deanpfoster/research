import numpy as np

def run_convex_sim(n, gamma, symmetric=False):
    d = 10
    eta = 0.01
    H = np.diag(np.linspace(0.1, 1.0, d))
    steps = 10000
    
    x = np.random.randn(n, d) * 10.0
    
    for t in range(steps):
        if symmetric:
            S = np.random.choice([-1, 1], size=(n, n))
            S = np.triu(S, 1)
            S = S + S.T
            np.fill_diagonal(S, 0)
        else:
            S = np.random.choice([-1, 1], size=(n, n))
            np.fill_diagonal(S, 0)
            
        g = x @ H
        noise = np.random.randn(n, d) * 1.0
        
        diff = x[None, :, :] - x[:, None, :]
        force = gamma * np.einsum('ij,ijk->ik', S, diff)
        
        x = x - eta * g + force + eta * noise
        
    mean_x = np.mean(x, axis=0)
    loss_of_mean = 0.5 * np.sum(mean_x * (mean_x @ H))
    avg_loss = 0.5 * np.mean(np.sum(x * (x @ H), axis=1))
    return loss_of_mean, avg_loss

print("Uncoupled:", run_convex_sim(20, 0.0))
print("Asymmetric:", run_convex_sim(20, 0.005, symmetric=False))
print("Symmetric:", run_convex_sim(20, 0.005, symmetric=True))
print("Asymmetric high:", run_convex_sim(20, 0.01, symmetric=False))
print("Symmetric high:", run_convex_sim(20, 0.01, symmetric=True))
