import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# PARAMETERS
# -------------------------

N = 15
alpha = 0.05
steps = 300
R = 0.5
goal = np.array([8.0, 8.0])

# -------------------------
# INITIAL POSITIONS
# -------------------------

positions = np.random.rand(N, 2) * 2
positions_prev = positions.copy()

history = [positions.copy()]

# -------------------------
# GOAL GRADIENT
# -------------------------

def grad_goal(x):
    return x - goal

# -------------------------
# WALL GRADIENT (simple box walls)
# -------------------------

def grad_walls(x):
    grad = np.zeros(2)
    wall_strength = 5.0
    
    if x[0] < 0:
        grad[0] += -wall_strength * x[0]
    if x[0] > 10:
        grad[0] += wall_strength * (x[0] - 10)
    if x[1] < 0:
        grad[1] += -wall_strength * x[1]
    if x[1] > 10:
        grad[1] += wall_strength * (x[1] - 10)
        
    return grad

# -------------------------
# ISOTROPIC SOCIAL FORCE
# -------------------------

def grad_social_iso(xi, xj):
    d_vec = xi - xj
    d = np.linalg.norm(d_vec)
    
    if d < R and d > 1e-6:
        return (R - d) * (d_vec / d)
    else:
        return np.zeros(2)

# -------------------------
# ANISOTROPIC SOCIAL FORCE
# -------------------------

def grad_social_ani(xi, xj, vi, beta=2.0):
    d_vec = xi - xj
    d = np.linalg.norm(d_vec)
    
    if d < R and d > 1e-6:
        v_norm = np.linalg.norm(vi)
        if v_norm > 1e-6:
            v_hat = vi / v_norm
            direction = (xj - xi) / d
            weight = 1 + beta * max(0, np.dot(v_hat, direction))
        else:
            weight = 1
            
        return weight * (R - d) * (d_vec / d)
    else:
        return np.zeros(2)

# -------------------------
# SIMULATION LOOP
# -------------------------

use_anisotropic = False  # Change to True for anisotropic

for k in range(steps):
    
    new_positions = positions.copy()
    
    for i in range(N):
        xi = positions[i]
        grad = grad_goal(xi) + grad_walls(xi)
        
        for j in range(N):
            if i != j:
                if use_anisotropic:
                    vi = positions[i] - positions_prev[i]
                    grad += grad_social_ani(xi, positions[j], vi)
                else:
                    grad += grad_social_iso(xi, positions[j])
        
        new_positions[i] = xi - alpha * grad
    
    positions_prev = positions.copy()
    positions = new_positions.copy()
    history.append(positions.copy())

history = np.array(history)

# -------------------------
# ANIMATION
# -------------------------

fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
scat = ax.scatter(history[0,:,0], history[0,:,1])

def update(frame):
    scat.set_offsets(history[frame])
    return scat,

ani = FuncAnimation(fig, update, frames=len(history), interval=30)
plt.show()
