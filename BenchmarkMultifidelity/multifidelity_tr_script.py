import matplotlib.pyplot as plt
import scipy.optimize
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

def f_lo_adjusted(x, z, grad_high_z, low_fidelity_loss_z, high_fidelity_loss_z):
    return low_fidelity_loss(x) + (high_fidelity_loss_z - low_fidelity_loss_z) + grad_high_z @ (x - z)

def step_loss(s, z, grad_high_z, low_fidelity_loss_z, high_fidelity_loss_z):
    return f_lo_adjusted(z + s, z, grad_high_z, low_fidelity_loss_z, high_fidelity_loss_z)

def grad_high(x):
    grad_hf_loss = grad(high_fidelity_loss)
    return grad_hf_loss(x)

#z0: initial point
#delta_max: max tr radius
#delta: current tr radius
#eta1,2: scaling for trust region, increase or decrease (make dynamic?)
#gamma1,2: thresholds for ratio of actual vs predicted reduction
#tol: termination value for minimum actual reduction
#loss_req: termination value for actual loss
def trust_region_optimization(z0, delta_max=0.2, eta1=0.25, eta2=0.75, gamma1=0.1, gamma2=0.75, max_iter=30, tol=1e-6, loss_req = 0.125):
    z = z0
    iteration = 0
    delta = delta_max * 0.1
    history = [z0.copy()]
    obj_high_values = [high_fidelity_loss(z)]
    gradient_norms = [np.linalg.norm(grad_high(z))]


    while iteration < max_iter:
        print(f"Iteration {iteration}:")
        print(f"  Current point z = {z}, Delta = {delta:.10f}")

        high_fidelity_loss_z = high_fidelity_loss(z)
        low_fidelity_loss_z = low_fidelity_loss(z)
        grad_high_z = grad_high(z)

        bounds = [(-delta, delta) for _ in z]
        res = scipy.optimize.minimize(
            step_loss,
            np.zeros_like(z),
            args=(z, grad_high_z, low_fidelity_loss_z, high_fidelity_loss_z),
            bounds=bounds,
            method="L-BFGS-B"
        )
        s = res.x
        print(f"  Step s = {s}")

        actual_reduction = high_fidelity_loss(z) - high_fidelity_loss(z + s)
        predicted_reduction = high_fidelity_loss(z) - f_lo_adjusted(z + s, z, grad_high_z, low_fidelity_loss_z, high_fidelity_loss_z)
        gamma = actual_reduction / predicted_reduction if predicted_reduction > 0 else 0

        print(f"  f_high(z) = {high_fidelity_loss(z):.6f}, f_high(z + s) = {high_fidelity_loss(z + s):.6f}")
        print(f"  Actual reduction = {actual_reduction:.6f}, Predicted reduction = {predicted_reduction:.6f}")
        print(f"  Gamma = {gamma:.4f}")

        if high_fidelity_loss(z) < loss_req:
            break

        if gamma < gamma1:
            delta = max(eta1 * delta, tol)
        elif gamma > gamma2:
            delta = min(eta2 * delta, delta_max)

        if actual_reduction > 0:
            z = z + s
            history.append(z.copy())
            obj_high_values.append(high_fidelity_loss(z))
            gradient_norms.append(np.linalg.norm(grad_high(z)))

        print(f"  Updated Delta = {delta:.10f}")
        print(f"  Gradient norm = {gradient_norms[-1]:.6f}")

        if gradient_norms[-1] < tol:
            print("  Convergence achieved.")
            break

        iteration += 1

    return z, obj_high_values, gradient_norms, history

z0 = np.array([-0.05, 0.02])
z_opt, obj_high_values, gradient_norms, history = trust_region_optimization(
    z0, delta_max=0.2, eta1=0.25, eta2=0.75, gamma1=0.1, gamma2=0.75, max_iter=15, tol=1e-6
)

D = 100
x1 = np.linspace(-0.2, 0.2, D)
x2 = np.linspace(-0.2, 0.2, D)
X1, X2 = np.meshgrid(x1, x2)
vectorized_loss = jax.vmap(lambda xy: high_fidelity_loss(xy), in_axes=0)

grid_points = np.stack([X1.ravel(), X2.ravel()], axis=-1)

Z = vectorized_loss(grid_points).reshape(X1.shape)

plt.ion()

plt.figure(figsize=(12, 10))
contour = plt.contour(X1, X2, Z, levels=150, cmap='coolwarm')
plt.colorbar(contour, label="High-fidelity Loss Value")
history_array = np.array(history)
plt.plot(history_array[:, 0], history_array[:, 1], 'o--', color='yellow', label="Optimization Path")
plt.scatter(history_array[0, 0], history_array[0, 1], color='green', label="Start Point", s=100)
plt.scatter(history_array[-1, 0], history_array[-1, 1], color='black', label="End Point", s=100)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Trust Region Optimization on Loss Surface")
plt.legend()
plt.grid()

# Plot optimization progress
plt.figure(figsize=(12, 6))
plt.plot(obj_high_values, label="High-fidelity Loss")
plt.plot(gradient_norms, label="Gradient Norm")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Trust Region Optimization Progress")
plt.legend()
plt.grid()
plt.show()

print("\nFinal Results:")
print(f"Optimal point: z = {z_opt}")
print(f"Optimal high-fidelity value: f_high(z) = {high_fidelity_loss(z_opt)}")


plt.pause(0)