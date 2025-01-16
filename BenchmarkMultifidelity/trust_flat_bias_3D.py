import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Rotation3D(theta, phi):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta * cos_phi, cos_theta * cos_phi, -sin_phi],
        [sin_theta * sin_phi, cos_theta * sin_phi, cos_phi]
    ])

def f1(x, x_star=np.array([0.1, 0.1, 0.1]), theta=0.2, phi=0.1):
    z = Rotation3D(theta, phi) @ (x - x_star)
    return np.sum(z**2 + 1 - np.cos(10 * np.pi * z))

iteration = 0
objective_value = []
location_step = []

def high_fidelity_loss(x):
    return f1(x)

def callback(x, res):
    global iteration, objective_value, location_step

    hf_loss = high_fidelity_loss(x)
    objective_value.append(hf_loss)
    location_step.append(x)

    print(f"Iteration {iteration}:")
    print(f"  Location: {x}")
    print(f"  HF Loss: {hf_loss}")

    iteration += 1

x0 = np.array([0.01, 0.08, 0.05])

bounds = [(-0.1, 0.2), (-0.1, 0.2), (-0.1, 0.2)]
linear_bounds = scipy.optimize.Bounds(
    [b[0] for b in bounds],
    [b[1] for b in bounds]
)

result = scipy.optimize.minimize(
    high_fidelity_loss,
    x0,
    method='trust-constr',
    bounds=linear_bounds,
    callback=callback,
    options={
        'xtol': 1e-8,
        'gtol': 1e-6,
        'initial_tr_radius': 0.1,
        'maxiter': 100
    }
)

print("\nOptimization Results:")
print(f"Optimal Solution: {result.x}")
print(f"Optimal Loss: {result.fun}")

D = 25
x1 = np.linspace(-0.1, 0.2, D)
x2 = np.linspace(-0.1, 0.2, D)
x3 = np.linspace(-0.1, 0.2, D)

X1, X2, X3 = np.meshgrid(x1, x2, x3)
Z = np.array([f1(np.array([x1, x2, x3])) for x1, x2, x3 in zip(X1.flatten(), X2.flatten(), X3.flatten())])
Z = Z.reshape(X1.shape)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(X1[::2, ::2, ::2], X2[::2, ::2, ::2], X3[::2, ::2, ::2], c=Z[::2, ::2, ::2], cmap='viridis', s=80, alpha=0.2)

location_array = np.array(location_step)
if location_array.size > 0:
    ax.plot(location_array[:, 0], location_array[:, 1], location_array[:, 2], color='red', linewidth=2, label='Optimization Path')
    ax.scatter(location_array[:, 0], location_array[:, 1], location_array[:, 2], color='red', s=20, label='Steps')

ax.scatter(x0[0], x0[1], x0[2], color='green', s=50, label='START')
ax.scatter(result.x[0], result.x[1], result.x[2], color='black', s=50, label='END')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('3D Optimization Landscape and Path')
fig.colorbar(sc, ax=ax, label='Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(objective_value, label="Objective Value")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss During Optimization")
plt.grid()
plt.legend()
plt.show()
