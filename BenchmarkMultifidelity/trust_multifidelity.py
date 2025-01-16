import os
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def Rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def f1(x, x_star=np.array([0.1, 0.1]), theta=0.2):
    z = Rotation(theta) * (x - x_star)
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
first_iter = True

def high_fidelity_loss(x):
    return f1(x) + e_r(x, phi=10000)

def low_fidelity_loss(x):
    return f1(x) + e_r(x, phi=2500)

def multifidelity_loss(x):
    global bias, first_iter
    
    if first_iter:
        hf_loss = high_fidelity_loss(x)
        lf_loss = low_fidelity_loss(x)
        bias = hf_loss - lf_loss

        mf_loss = lf_loss + bias

        print(f"Initial Eval: ")
        print(f"  HF Loss: {hf_loss}")
        print(f"  LF Loss: {lf_loss}")
        print(f"  Bias: {bias}")
        print(f"  MF Loss: {mf_loss}")

        first_iter = False

    else:
        lf_loss = low_fidelity_loss(x)
        mf_loss = lf_loss + bias
        print(f"    FunctionEval: LF Loss = {lf_loss}, Bias = {bias}, MF Loss = {mf_loss}")
        
    return mf_loss

def callback(x, res):
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

x0 = np.array([0.01, 0.095])

bounds = [(-0.1, 0.2), (-0.1, 0.2)]
linear_bounds = scipy.optimize.Bounds(
    [b[0] for b in bounds],
    [b[1] for b in bounds]
)

result = scipy.optimize.minimize(
    multifidelity_loss,
    x0,
    method='trust-constr',
    bounds=linear_bounds,
    callback=callback,
    options={
        'xtol': 1e-8,
        'gtol': 1e-6,
        'initial_tr_radius': 0.1,
        'maxiter': 200
    }
)

print("\nOptimization Results:")
print(f"Optimal Solution: {result.x}")
print(f"Optimal Loss: {result.fun}")

D = 100
x1 = np.linspace(-0.1, 0.2, D)
x2 = np.linspace(-0.1, 0.2, D)
X1, X2 = np.meshgrid(x1, x2)
Z = np.array([[f1(np.array([x1, x2])) for x1, x2 in zip(row1, row2)] for row1, row2 in zip(X1, X2)])

plt.figure()
contour = plt.contour(X1, X2, Z, cmap='coolwarm', levels=200)
plt.colorbar(contour)
plt.plot(x0[0], x0[1], '.', color='green', ms=10, label='START')
location_array = np.array(location_step)
if location_array.size > 0:
    plt.plot(location_array[:, 0], location_array[:, 1], '--', color='yellow', label='Path')
    plt.plot(location_array[:, 0], location_array[:, 1], '.', color='yellow', ms=10)
plt.plot(result.x[0], result.x[1], '.', color='black', ms=10, label='END')
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Optimization Path on Function Contour')
plt.show()

plt.figure()
plt.plot(objective_value, label="Multifidelity Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss During Optimization")
plt.grid()
plt.legend()
plt.show()
