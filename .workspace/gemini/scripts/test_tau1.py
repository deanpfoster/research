import numpy as np

def run_tau1_sim():
    N = 20
    d = 2
    eta = 0.01
    gamma = 0.05
    steps = 10000
    
    # Bump params
    c = np.array([10.0, 0.0])
    r = 1.0
    lmbda = 0.01
    
    # Let's try b = 1.0, 10.0, 100.0, 1000.0
    for b in [1.0, 10.0, 100.0, 1000.0]:
        x = np.random.randn(N, d) * 0.1
        x[0] = c + np.random.randn(d) * 0.1 # Agent 0 in the bump
        
        escaped_bump = False
        found_bump = False
        
        for t in range(steps):
            S = np.random.choice([-1, 1], size=(N, N))
            np.fill_diagonal(S, 0)
            
            # Gradients (ascent on f)
            g = -2 * lmbda * x
            
            # Bump gradient
            for i in range(N):
                dist_sq = np.sum((x[i] - c)**2)
                q = 1.0 + dist_sq / (r**2)
                bump_g = -b * 2.0 * (x[i] - c) / (r**2 * q**2)
                g[i] += bump_g
                    
            diff = x[None, :, :] - x[:, None, :]
            force = gamma * np.einsum('ij,ijk->ik', S, diff)
            
            x = x + eta * g + force
            
            if np.linalg.norm(x[0] - c) > 3.0 and not escaped_bump:
                escaped_bump = True
                escape_time = t
                
            if not found_bump:
                dists = np.linalg.norm(x[1:] - c[None, :], axis=1)
                if np.any(dists < 1.0):
                    found_bump = True
                    found_time = t
                    
            if escaped_bump and found_bump:
                break
                
        print(f"b = {b}:")
        if found_bump:
            print(f"  Exploring agent found bump at step {found_time}")
        else:
            print(f"  Exploring agents NEVER found bump.")
            
        if escaped_bump:
            print(f"  Anchored agent ESCAPED bump at step {escape_time}")
        else:
            print(f"  Anchored agent STAYED in bump.")

run_tau1_sim()
