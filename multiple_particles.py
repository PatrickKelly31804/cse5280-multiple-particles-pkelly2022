import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 24
alpha = 0.04
steps = 450

R = 0.70
beta = 10.0

use_anisotropic = True

goal_left = np.array([1.0, 5.0])
goal_right = np.array([9.0, 5.0])

positions = np.zeros((N, 2))

for i in range(N // 2):
    positions[i] = np.array([1.2, 3.0 + (i / (N//2 - 1)) * 4.0])
for i in range(N // 2, N):
    j = i - (N // 2)
    positions[i] = np.array([8.8, 3.0 + (j / (N//2 - 1)) * 4.0])

positions_prev = positions.copy()
history = [positions.copy()]

def grad_goal_for(i, x):
    g = goal_right if i < N // 2 else goal_left
    return x - g

def grad_walls(x):
    grad = np.zeros(2)
    w = 18.0

    if x[0] < 0:   grad[0] += -w * x[0]
    if x[0] > 10:  grad[0] +=  w * (x[0] - 10)
    if x[1] < 0:   grad[1] += -w * x[1]
    if x[1] > 10:  grad[1] +=  w * (x[1] - 10)

    if x[1] < 3:
        grad[1] += -w * (x[1] - 3)
    if x[1] > 7:
        grad[1] +=  w * (x[1] - 7)

    if 4.6 < x[0] < 5.4:
        if x[1] < 4.3:
            grad[1] += -w * (x[1] - 4.3)
        if x[1] > 5.7:
            grad[1] +=  w * (x[1] - 5.7)

    return grad

def grad_social_iso(xi, xj):
    d_vec = xi - xj
    d = np.linalg.norm(d_vec)
    if d < R and d > 1e-6:
        return (R - d) * (d_vec / d)
    return np.zeros(2)

def grad_social_ani(xi, xj, vi):
    d_vec = xi - xj
    d = np.linalg.norm(d_vec)
    if d < R and d > 1e-6:
        v_norm = np.linalg.norm(vi)
        if v_norm > 1e-6:
            v_hat = vi / v_norm
            direction = (xj - xi) / d
            weight = 1 + beta * max(0.0, np.dot(v_hat, direction))
        else:
            weight = 1.0
        return weight * (R - d) * (d_vec / d)
    return np.zeros(2)

for k in range(steps):
    new_positions = positions.copy()

    for i in range(N):
        xi = positions[i]
        grad = grad_goal_for(i, xi) + grad_walls(xi)

        for j in range(N):
            if i == j:
                continue
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

trail_len = 30

fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("ANISOTROPIC" if use_anisotropic else "ISOTROPIC")

scat = ax.scatter(history[0, :, 0], history[0, :, 1])
lines = [ax.plot([], [])[0] for _ in range(N)]

def update(frame):
    pts = history[frame]
    scat.set_offsets(pts)

    start = max(0, frame - trail_len)
    for i in range(N):
        tr = history[start:frame+1, i, :]
        lines[i].set_data(tr[:, 0], tr[:, 1])

    return [scat] + lines

ani = FuncAnimation(fig, update, frames=len(history), interval=25)
plt.show()
