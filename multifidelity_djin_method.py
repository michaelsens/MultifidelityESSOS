from ESSOS import Coils, Particles, loss, CreateEquallySpacedCurves

import os
import jax
import sys
import pybobyqa # type: ignore
from jax import jit
from time import time
import jax.numpy as jnp
from matplotlib import cm
from functools import partial
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from scipy.optimize import least_squares, minimize

@jit
def loss_partial_dofs_min(x):
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    return loss_partial(dofs, currents)

def high_fidelity_loss(dofs, dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model):
    return loss(dofs, dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)

def low_fidelity_loss(dofs, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model):
    return loss(dofs, dofs_currents, coils, reduced_particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)


iteration = 0
bias = 0
is_high_fidelity_step = True
curr_high_fidelity_val = 0
is_first_iter = True
base_distance = 0
tol = 0 #need to maybe figure out how to change tolerance to be the based on the consistancy of the bias rather than its absolute value, this seems complicated to implement
biases = []

def multifidelity_loss(x):
    global bias, is_high_fidelity_step, curr_high_fidelity_val, base_distance, is_first_iter, tol
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    
    if is_high_fidelity_step:
        high_fidelity_value = high_fidelity_loss(dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)

        curr_high_fidelity_val = high_fidelity_value

        if is_first_iter:
            low_fidelity_value = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
            bias = high_fidelity_value - low_fidelity_value
            biases.append(bias)
            tol = abs(bias) / 5
            print("Starting Bias:", bias)
            print("Starting Tol: ", tol)
            is_first_iter = False

        return high_fidelity_value
    else:
        low_fidelity_value = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        return low_fidelity_value + bias

def callback(x):
    global iteration, is_high_fidelity_step, bias, objective_value, curr_high_fidelity_val, base_distance, tol

    objective_value += [loss_partial_dofs_min(x)]
    print("Objective Function: {}".format(loss_partial_dofs_min(x)))

    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    
    if is_high_fidelity_step:
        high_fidelity_value = high_fidelity_loss(dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        low_fidelity_value = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        bias = high_fidelity_value - low_fidelity_value

        if bias > base_distance*10: # this approach is maybe wrong? Im starting to question weather exact bias even matters rather we need variables to track the variance of the bias ratio from iteration to iteration
            pass # Not sure how to implement backtracking code

        if abs(bias - biases[-1]) < tol:
            is_high_fidelity_step = False

        biases.append(bias)
        print(f"HighFidelity Step: HF Loss {high_fidelity_value} --- LF Loss {low_fidelity_value} --- Bias {bias}")
        print("------------")
    else:
        low_fidelity_value = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        if abs((curr_high_fidelity_val - low_fidelity_value) - biases[-1]) > tol:
            #tol = tol * (base_distance / abs(curr_high_fidelity_val - low_fidelity_value)) # tolerance adjustment based on ratio
            print("Adjusted Tolerance: ", tol)
            is_high_fidelity_step = True

        print(f"LowFidelity Step: LF Loss {low_fidelity_value} --- Current Error: {curr_high_fidelity_val - low_fidelity_value}") 
        print("-----------")
    
    iteration += 1
    
#### INPUT PARAMETERS START HERE - NUMBER OF PARTICLES ####
number_of_cores = 3
number_of_particles_per_core = 5
#### Some other imports
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_cores}'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.insert(1, os.path.dirname(os.getcwd()))
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, loss
from MagneticField import B, B_norm
#### Input parameters continue here
n_curves=2
nfp=4
order=2
r = 3
A = 2 # Aspect ratio
R = A*r
r_init = r/4
maxtime = 3.0e-6
timesteps_guiding_center=max(1000,int(maxtime/1.0e-8))
timesteps_lorentz = 1000
#timesteps_lorentz=int(maxtime/1.0e-10)
nparticles = number_of_cores*number_of_particles_per_core
n_segments=80
coil_current = 7e6
change_currents = False
model = 'Lorentz' # 'Guiding Center' or 'Lorentz'
method = 'least_squares' # 'least_squares','L-BFGS-B','Bayesian','BOBYQA', or one of scipy.optimize.minimize methods such as 'BFGS'
max_function_evaluations = 5
max_iterations_BFGS = 100
max_function_evaluations_BFGS = 400
max_function_evaluations_BOBYQA = 550
tolerance_to_terminace_optimization = 1e-6
min_val = -11 # minimum coil dof value
max_val =  15 # maximum coil dof value
##### Input parameters stop here
particles = Particles(nparticles)

reduced_particles = Particles(max(1, nparticles // 5))
curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([coil_current]*n_curves))
timesteps = timesteps_lorentz if model=='Lorentz' else timesteps_guiding_center

x0, y0, z0, vpar0, vperp0 = stel.initial_conditions(particles, R, r_init, model='Guiding Center')
v0 = jnp.zeros((3, nparticles))
for i in range(nparticles):
    B0 = B(jnp.array([x0[i],y0[i],z0[i]]), curve_segments=stel.gamma(n_segments), currents=stel.currents)
    b0 = B0/jnp.linalg.norm(B0)
    perp_vector_1 = jnp.array([0, b0[2], -b0[1]])
    perp_vector_1_normalized = perp_vector_1/jnp.linalg.norm(perp_vector_1)
    perp_vector_2 = jnp.cross(b0, perp_vector_1_normalized)
    v0 = v0.at[:,i].set(vpar0[i]*b0 + vperp0[i]*(perp_vector_1_normalized/jnp.sqrt(2)+perp_vector_2/jnp.sqrt(2)))
normB0 = jnp.apply_along_axis(B_norm, 0, jnp.array([x0, y0, z0]), stel.gamma(n_segments), stel.currents)
μ = particles.mass*vperp0**2/(2*normB0)
start = time()
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init, jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]), maxtime, timesteps, n_segments, model=model)
print(f"Loss function initial value: {loss_value:.8f}")
end = time()
print(f"Took: {end-start:.2f} seconds")

initial_values = jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]) if model=='Lorentz' else jnp.array([x0, y0, z0, vpar0, vperp0])

iteration = 0
loss_vals = []



# Loss partial function
loss_partial = partial(loss, old_coils=stel, particles=particles, R=R, r_init=r_init, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=n_segments, model=model)

len_dofs = len(jnp.ravel(stel.dofs))
if change_currents:
    all_dofs = jnp.concatenate((jnp.ravel(stel.dofs), jnp.ravel(stel.currents)[:n_curves]))
else:
    all_dofs = jnp.ravel(stel.dofs)

# Optimization
print(f"Number of dofs: {len(all_dofs)}")
start_optimization_time = time()

objective_value = []
objective_value += [loss_value]

res = minimize(multifidelity_loss, x0=all_dofs, method='L-BFGS-B', options={ 'maxiter': max_iterations_BFGS, 'maxcor': 300},callback = callback, tol=1e-15)

x = jnp.array(res.x)
end_optimization_time = time()
optimization_duration = end_optimization_time - start_optimization_time
print(f"Optimization took: {optimization_duration:.2f} seconds")
print(f"Resulting dofs: {repr(x.tolist())}")

# Plotting loss values
plt.plot(objective_value, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()