import numpy as np
import scipy
import matplotlib.pyplot as plt

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

iteration = 0
bias = 0
objective_value = []
location_step = []
sampled_points = []
target_iteration = 0

def high_fidelity_loss(x):
    return f1(x) + e_r(x, phi=10000)

def low_fidelity_loss(x):
    return f1(x) + e_r(x, phi=2500)

def loss(x):
    global bias, iteration, sampled_points
    lf_loss = low_fidelity_loss(x)
    hf_loss = high_fidelity_loss(x)
    coords = [f"{coord:.2f}" for coord in x]
    print(f"    FunctionEval: Loss = {lf_loss:.6f}, Coordinates = {coords}")
    return lf_loss

def callback(x):
    global iteration, objective_value, location_step
    hf_loss = high_fidelity_loss(x)
    lf_loss = low_fidelity_loss(x)
    objective_value.append(hf_loss)
    location_step.append(x.copy())
    coords = [f"{coord:.2f}" for coord in x]
    print(f"Iteration {iteration}: HF Loss = {hf_loss:.6f}, LF Loss = {lf_loss:.6f}, Coordinates = {coords}")
    iteration += 1

def custom_jacobian(x):
    global sampled_points
    r = 0.05
    gradient = np.zeros_like(x)  # Initialize gradient array
    for i in range(len(x)):
        perturbation = np.zeros_like(x)
        perturbation[i] = r  # Positive perturbation in the i-th dimension
        positive_point = x + perturbation
        perturbation[i] = -r  # Negative perturbation in the i-th dimension
        negative_point = x + perturbation
        grad_i = (low_fidelity_loss(positive_point) - low_fidelity_loss(negative_point)) / (2 * r)
        gradient[i] = grad_i
        if iteration == target_iteration:
            sampled_points.append(positive_point.copy())
            sampled_points.append(negative_point.copy())
    return gradient


x0 = np.array([0.01, 0.08])
bounds = [(-0.1, 0.2), (-0.1, 0.2)]

result = scipy.optimize.minimize(
    loss,
    x0,
    method='L-BFGS-B',
    jac=custom_jacobian,
    bounds=bounds,
    callback=callback,
    options={'ftol': 1e-12, 'gtol': 1e-8, 'eps': 1e-9, 'maxiter': 100}
)

print("\nOptimization Results:")
print(f"Optimal Solution: {result.x}")
print(f"Optimal Loss: {result.fun}")

D = 100
x1 = np.linspace(-0.1, 0.2, D)
x2 = np.linspace(-0.1, 0.2, D)
X1, X2 = np.meshgrid(x1, x2)
Z = np.array([[f1(np.array([x1, x2])) for x1, x2 in zip(row1, row2)] for row1, row2 in zip(X1, X2)])

plt.figure(figsize=(12, 10))
contour = plt.contour(X1, X2, Z, cmap='coolwarm', levels=50)
plt.colorbar(contour, label="Loss Value")
plt.plot(x0[0], x0[1], 'o', color='green', ms=10, label='Start Point')
plt.plot(np.array(location_step)[:, 0], np.array(location_step)[:, 1], 'x--', color='yellow', label='Optimization Path')
plt.plot(result.x[0], result.x[1], 'o', color='black', ms=10, label='End Point')
if len(sampled_points) > 0:
    sampled_array = np.array(sampled_points)
    plt.scatter(sampled_array[:, 0], sampled_array[:, 1], color='red', s=20, label=f'Sampled Points (Iteration {target_iteration})')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Optimization Landscape and Sampled Points")
plt.legend()
plt.grid()

print(f"\nSampled Points for Iteration {target_iteration}:")
if len(sampled_points) > 0:
    for i, point in enumerate(sampled_points, start=1):
        print(f"  Point {i}: {point}")
else:
    print("  No points sampled during the target iteration.")

plt.figure(figsize=(10, 6))
plt.plot(objective_value, marker='o', label="Multifidelity Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Over Iterations")
plt.grid()
plt.legend()
plt.show()
