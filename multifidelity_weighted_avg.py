from ESSOS import Coils, Particles, loss, CreateEquallySpacedCurves, Curves

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

@partial(jit, static_argnums=(2, 3, 4, 5, 7, 8, 9, 10))
def particleloss(dofs:           jnp.ndarray,
         dofs_currents:  jnp.ndarray,
         old_coils:      Coils,
         particles:      Particles,
         R:              float,
         r_init:         float,
         initial_values: jnp.ndarray,
         maxtime:        float,
         timesteps:      int,
         n_segments:     int,
         model:          str = 'Guiding Center') -> float:
             
    """ Loss function to be minimized
        Attributes:
    dofs: Fourier Coefficients of the independent coils - shape (n_indcoils*3*(2*order+1)) - must be a 1D array
    dofs_currents: Currents of the independent coils - shape (n_indcoils,)
    old_coils: Coils from which the dofs and dofs_currents are taken
    n_segments: Number of segments to divide each coil
    particles: Particles to optimize the trajectories
    maxtime: Maximum time of the simulation
    timesteps: Number of timesteps
    initial_values: Initial values of the particles - shape (5, n_particles)
    R: float: Major radius of the loss torus
        Returns:
    loss_value: Loss value - must be scalar
    """

    n_indcoils = jnp.size(old_coils.dofs, 0)
    nfp = old_coils.nfp
    stellsym = old_coils.stellsym

    #dofs = jnp.reshape(dofs, (n_indcoils, 3, -1))
    dofs = jnp.reshape(jnp.array([[[5.931931454615609, 0.11127878241909184, 1.9936042669464322, 0.08086991423720513, -0.0007190035397465135, 0.05570068949850574, 0.13823038792286735], [1.208997674963824, 0.06380884104454711, 0.42373981592319193, -0.13759370252270314, 0.1694989165215695, 0.10123839997922662, 0.25872475114609444], [0.2119190131809971, -1.92737798715156, -0.25085811031090566, -0.04151071988165265, -0.13285535949148117, 0.06765643669911842, 0.34728143230671876]], [[4.878130239841212, -0.006396477988941415, 1.4097000390785464, -0.03336214189728549, 0.2776620077045304, -0.05147026377047189, 0.01087658778127479], [3.5412430351136415, 0.009319533558537001, 1.2595626361386478, -0.024633086847271134, -0.05014236644443539, 0.07498232418702869, -0.014229582498650823], [0.0676012063968044, -1.836171647573297, -0.03921559289799803, 0.028383896665587743, -0.04908650840493748, 0.4189707698658851, 0.02905434339315262]]]), (n_curves, 3, 2*order+1))
    curves = Curves(dofs, nfp=4, stellsym=True)

    coils = Coils(curves, jnp.array([coil_current]*n_curves))

    if model=='Guiding Center':
        trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, n_segments)
    elif model=='Lorentz':
        trajectories = coils.trace_trajectories_lorentz(particles, initial_values, maxtime, timesteps, n_segments)
    else:
        raise ValueError("Model must be 'Guiding Center' or 'Lorentz'")
    
    distances_squared = jnp.square(
        jnp.sqrt(
            trajectories[:, :, 0]**2 + trajectories[:, :, 1]**2
        )-R
    )+trajectories[:, :, 2]**2

    particle_losses = jnp.mean(distances_squared, axis=1) / r_init**2
    
    return jnp.mean(particle_losses), particle_losses

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
    return particleloss(dofs, dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)

def low_fidelity_loss(dofs, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model):
    return loss(dofs, dofs_currents, coils, reduced_particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)


iteration = 0
bias = 0
first_iter = True
idxs = []
weight = 0.99
high_fidelity_value = 0

def multifidelity_loss(x):
    global high_fidelity_value, first_iter, idxs
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    if first_iter:
        high_fidelity_value, particle_losses = high_fidelity_loss(dofs, currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)

        idxs = jnp.argsort(particle_losses)[-nparticles // 3:]
        print(idxs)
        print(initial_values)
        low_fidelity_value = low_fidelity_loss(dofs, currents, stel, R, r_init, initial_values[:,idxs], maxtime, timesteps, n_segments, model)
        print(f"HighFidelity Step: HF Loss {high_fidelity_value} --- LF Loss {low_fidelity_value}")
        first_iter = False
    else:
        low_fidelity_value = low_fidelity_loss(dofs, currents, stel, R, r_init, initial_values[:,idxs], maxtime, timesteps, n_segments, model)
        #print(f"LF Loss {low_fidelity_value} --- Bias {bias} --- Loss: {low_fidelity_value + bias}")
    
    return (1 - weight) * low_fidelity_value + weight * high_fidelity_value

def callback(x):
    global flag_value,iteration, bias, objective_value, high_fidelity_value

    objective_value += [multifidelity_loss(x)]
    print("callback")

    print(f"Iteration {iteration}:")
    print("Objective Function: {}".format(multifidelity_loss(x)))

    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    high_fidelity_value, particle_losses = high_fidelity_loss(dofs, currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)

    idxs = jnp.argsort(particle_losses)[-nparticles // 3:]

    print(idxs)

    low_fidelity_value = low_fidelity_loss(dofs, currents, stel, R, r_init, initial_values[:,idxs], maxtime, timesteps, n_segments, model)
    print(f"HighFidelity Step: HF Loss {high_fidelity_value} --- LF Loss {low_fidelity_value}")

    iteration += 1

#### INPUT PARAMETERS START HERE - NUMBER OF PARTICLES ####
number_of_cores = 3
number_of_particles_per_core = 8
#### Some other imports
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_cores}'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.insert(1, os.path.dirname(os.getcwd()))
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, loss
from MagneticField import B, B_norm
#### Input parameters continue here
n_curves=2
nfp=4
order=3
r = 3
A = 2 # Aspect ratio
R = A*r
r_init = r/4
maxtime = 3.0e-6
timesteps_guiding_center=max(1000,int(maxtime/1.0e-8))
timesteps_lorentz=1000
#timesteps_lorentz=int(maxtime/1.0e-10)
nparticles = number_of_cores*number_of_particles_per_core
n_segments=100
coil_current = 7e6
change_currents = False
model = 'Lorentz' # 'Guiding Center' or 'Lorentz'
method = 'least_squares' # 'least_squares','L-BFGS-B','Bayesian','BOBYQA', or one of scipy.optimize.minimize methods such as 'BFGS'
max_function_evaluations = 5
max_iterations_BFGS = 1000
max_function_evaluations_BFGS = 400
max_function_evaluations_BOBYQA = 550
tolerance_to_terminace_optimization = 1e-6
min_val = -11 # minimum coil dof value
max_val =  15 # maximum coil dof value
##### Input parameters stop here
particles = Particles(nparticles)

reduced_particles = Particles(max(1, nparticles // 3))

print(particles)

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
flag_value = []


#res = minimize(multifidelity_loss, x0=all_dofs, method='L-BFGS-B', options={ 'maxiter': 1000,'disp': False, 'maxcor': 300, 'ftol': 1e-18, 'gtol': 1e-12, 'eps': 1e-8}, callback = callback, tol=1e-8)

while iteration < 15:
    
    if change_currents:
        all_dofs = jnp.concatenate((jnp.ravel(stel.dofs), jnp.ravel(stel.currents)[:n_curves]))
    else:
        all_dofs = jnp.ravel(stel.dofs)

    res = minimize(multifidelity_loss, x0=all_dofs, method='L-BFGS-B', options={ 'maxiter': 15,'disp': False, 'maxcor': 300, 'ftol': 1e-18, 'gtol': 1e-12, 'eps': 1e-8}, callback = callback, tol=1e-8)

x = jnp.array(res.x)
end_optimization_time = time()
optimization_duration = end_optimization_time - start_optimization_time
print(f"Optimization took: {optimization_duration:.2f} seconds")
print(f"Resulting dofs: {repr(x.tolist())}")

arg = jnp.argwhere(jnp.array([flag_value[1:]],dtype = float)==1)[:,1]

# Plotting loss values
plt.plot(objective_value, label='Loss')

for a in arg:
    plt.plot(a,objective_value[a], '.', color = 'red')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()