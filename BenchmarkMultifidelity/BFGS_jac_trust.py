import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Helper functions
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

# Global variables
iteration = 0
bias = 0
objective_value = []
location_step = []
evaluation_points = []  # For storing all points of evaluation
first_iter = True

# Loss functions
def high_fidelity_loss(x):
    return f1(x) + e_r(x, phi=10000)

def low_fidelity_loss(x):
    return f1(x) + e_r(x, phi=2500)

def multifidelity_loss(x):
    global bias, first_iter, evaluation_points
    lf_loss = low_fidelity_loss(x)
    if first_iter:
        hf_loss = high_fidelity_loss(x)
        bias = hf_loss - lf_loss
        first_iter = False
    mf_loss = lf_loss + bias
    evaluation_points.append(x)  # Record the evaluation point
    return mf_loss

# Callback function for optimization
def callback(x):
    global iteration, bias, objective_value, location_step
    hf_loss = high_fidelity_loss(x)
    lf_loss = low_fidelity_loss(x)
    bias = hf_loss - lf_loss
    mf_loss = lf_loss + bias
    objective_value.append(mf_loss)
    location_step.append(x)

    print(f"Iteration {iteration}:")
    print(f"  HF Loss: {hf_loss}")
    print(f"  LF Loss: {lf_loss}")
    print(f"  Bias: {bias}")
    print(f"  MF Loss: {mf_loss}")
    iteration += 1

# Initial conditions and bounds
x0 = np.array([0.01, 0.08])
bounds = [(-0.1, 0.2), (-0.1, 0.2)]

# Optimization process
result = scipy.optimize.minimize(
    multifidelity_loss,
    x0,
    method='L-BFGS-B',
    bounds=bounds,
    callback=callback,
    options={'ftol': 1e-12, 'gtol': 1e-8, 'eps': 1e-9, 'maxiter': 100}
)

# Results
print("Optimization Results:")
print(f"Optimal Solution: {result.x}")
print(f"Optimal Loss: {result.fun}")

# Generate data for contour plot
D = 100
x1 = np.linspace(-0.1, 0.2, D)
x2 = np.linspace(-0.1, 0.2, D)
X1, X2 = np.meshgrid(x1, x2)
Z = np.array([[f1(np.array([x1, x2])) for x1, x2 in zip(row1, row2)] for row1, row2 in zip(X1, X2)])

# Plot contour and points
plt.figure(figsize=(12, 8))
contour = plt.contour(X1, X2, Z, cmap='coolwarm', levels=20)
plt.colorbar(contour)
plt.plot(x0[0], x0[1], '.', color='green', ms=10, label='START')
plt.plot(result.x[0], result.x[1], '.', color='black', ms=10, label='END')
plt.plot(np.array(location_step)[:, 0], np.array(location_step)[:, 1], '--', color='yellow', label='Steps')
plt.scatter(*zip(*evaluation_points), color='blue', s=10, label='Evaluation Points')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

# Plot loss over iterations
plt.figure()
plt.plot(objective_value, label="Multifidelity Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Graph")
plt.grid()
plt.legend()
plt.show()
