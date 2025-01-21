import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as np
from jax import grad

def Rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def f1(x, x_star=np.array([0.1, 0.1]), theta=0.2):
    z = Rotation(theta) @ (x - x_star)
    return np.sum(z**2 + 1 - np.cos(10 * np.pi * z))

def e_r(x, x_star=np.array([0.1, 0.1]), theta=0.2, phi=10000):
    z = Rotation(theta) @ (x - x_star)
    O = 1 - 0.0001 * phi
    W = 10 * np.pi * O
    B = 0.5 * np.pi * O
    return np.sum(O * np.cos(W * z + B + np.pi)**2)

def high_fidelity_loss(x):
    return f1(x) + e_r(x, phi=10000)

def low_fidelity_loss(x):
    return f1(x) + e_r(x, phi=2500)

def compute_gradient(x):
    grad_loss = grad(low_fidelity_loss)
    return grad_loss(x)

def trust_region_optimizer(x0, max_iters=30, tol=1e-6, r_init=0.015, r_min=1e-5, r_max=0.1, decay_factor=0.9, loss_req = 0.125):
    x = x0
    iteration = 0
    objective_values = []
    location_steps = [x0.copy()]
    sampled_points = []
    r = r_init

    while iteration < max_iters:
        loss = low_fidelity_loss(x)
        objective_values.append(loss)
        gradient = compute_gradient(x)
        
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 1e-8:
            gradient = gradient / gradient_norm
        
        step = -gradient * r
        step_norm = np.linalg.norm(step)
        
        max_step = r * 2
        if step_norm > max_step:
            step *= max_step / step_norm

        x_new = x + step
        sampled_points.append(x_new.copy())

        print(f"Iteration {iteration}: Loss = {loss:.6f}, Step = {step}, Step Norm = {step_norm}, Radius = {r}, Location = {x_new}")
        
        if np.linalg.norm(step) < tol:
            break

        if loss < loss_req:
            break

        r = max(r_min, min(r_max, r * decay_factor))
        
        x = x_new
        location_steps.append(x.copy())
        iteration += 1

    return x, objective_values, location_steps, sampled_points

x0 = np.array([0.01, 0.038])
final_x, obj_values, steps, samples = trust_region_optimizer(x0)

D = 100
x1 = np.linspace(-0.1, 0.2, D)
x2 = np.linspace(-0.1, 0.2, D)
X1, X2 = np.meshgrid(x1, x2)
Z = np.array([[f1(np.array([x1, x2])) for x1, x2 in zip(row1, row2)] for row1, row2 in zip(X1, X2)])

plt.figure(figsize=(12, 10))
contour = plt.contour(X1, X2, Z, cmap='coolwarm', levels=150)
plt.colorbar(contour, label="Loss Value")
plt.plot(x0[0], x0[1], 'o', color='green', ms=10, label='Start Point')
steps_array = np.array(steps)
plt.plot(steps_array[:, 0], steps_array[:, 1], 'x--', color='yellow', label='Optimization Path')
plt.plot(final_x[0], final_x[1], 'o', color='black', ms=10, label='End Point')
samples_array = np.array(samples)
plt.scatter(samples_array[:, 0], samples_array[:, 1], color='red', s=20, label='Sampled Points')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Custom Trust-Region Optimization")
plt.legend()
plt.grid()

plt.figure(figsize=(10, 6))
plt.plot(obj_values, marker='o', label="Loss Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Over Iterations")
plt.grid()
plt.legend()
plt.show()
