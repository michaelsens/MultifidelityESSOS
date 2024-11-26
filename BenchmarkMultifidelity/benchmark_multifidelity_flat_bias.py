import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

def Rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# Define the shifted-rotated Rastrigin function
def f1(x, x_star=np.array([0.1, 0.1]), theta=0.2):
    z = Rotation(theta) * (x - x_star)
    return np.sum(z**2 + 1 - np.cos(10 * np.pi * z))

# adding resolution function 

def e_r(x, x_star=np.array([0.1, 0.1]), theta=0.2, phi = 10000): 

    z = Rotation(theta) * (x - x_star)

    O = 1 - .0001*phi
    W = 10 * np.pi * O
    B = .5 * np.pi * O
    
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

x0 = np.array([.01,.095])

bounds = [(-.1, .2), (-.1, .2)]

result = scipy.optimize.minimize(multifidelity_loss, x0, method='L-BFGS-B', bounds=bounds, callback=callback, options={'ftol': 1e-10, 'gtol': 1e-6, 'eps': 1e-8 })

print("Optimization Results:")
print(f"Optimal Solution: {result.x}")
print(f"Optimal Loss: {result.fun}")

D = 100

x1 = np.linspace(-1, 1, D)
x2 = np.linspace(-1, 1, D)
x1 = np.linspace(-.1, .2, D)
x2 = np.linspace(-.1, .2, D)


X1, X2 = np.meshgrid(x1, x2)
Z = np.array([[f1(np.array([x1, x2])) for x1, x2 in zip(row1, row2)] for row1, row2 in zip(X1, X2)])

np.shape(np.array([location_step])[0])

plt.figure()
contour = plt.contour(X1, X2, Z, cmap='coolwarm', levels = 200)
plt.colorbar(contour)
plt.plot(x0[0],x0[1],'.', color = 'green',ms = 10, label = 'START')
plt.plot(np.array([location_step])[0][:,0],np.array([location_step])[0][:,1],'--', color = 'yellow',ms = 10, label = 'Temp')
plt.plot(np.array([location_step])[0][:,0],np.array([location_step])[0][:,1],'.', color = 'yellow',ms = 10, label = 'Temp')
plt.plot(result.x[0], result.x[1],'.', color = 'black',ms = 10,label = 'END')
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')

plt.figure()
plt.plot(objective_value, label="Multifidelity Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Graph")
plt.grid()
plt.legend()
plt.show()
