from ESSOS import Coils, Particles, loss, CreateEquallySpacedCurves
import os
import jax
import sys
import pybobyqa  # type: ignore
from jax import jit
from time import time
import jax.numpy as jnp
import numpy as np
from matplotlib import cm
from functools import partial
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from scipy.optimize import least_squares, minimize

iteration = 0
bias = 0
first_iter = True
objective_value = []
flag_value = []

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

def multifidelity_loss(x):
    global bias, first_iter
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    if first_iter:
        hf = high_fidelity_loss(dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        lf = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        bias = hf - lf
        print(f"HighFidelity Step: HF Loss {hf} --- LF Loss {lf} --- Bias {bias}")
        first_iter = False
    else:
        lf = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    
    return lf

def callback(x, res=None):
    global flag_value, iteration, bias, objective_value

    print("callback")
    print(f"Iteration {iteration}:")
    current_loss = multifidelity_loss(x)
    print("Objective Function: {}".format(current_loss))

    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    objective_value.append(high_fidelity_loss(dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model))
    
    hf = high_fidelity_loss(dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    lf = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    bias = hf - lf
    print(f"HighFidelity Step: HF Loss {hf} --- LF Loss {lf} --- Bias {bias}")

    iteration += 1

#### INPUT PARAMETERS START HERE - NUMBER OF PARTICLES ####
number_of_cores = 3
number_of_particles_per_core = 8
#### Some other imports and configurations
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_cores}'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.insert(1, os.path.dirname(os.getcwd()))
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, loss
from MagneticField import B, B_norm
#### Input parameters continue here
n_curves = 2
nfp = 4
order = 2
r = 3
A = 2  # Aspect ratio
R = A * r
r_init = r / 4
maxtime = 3.0e-6
timesteps_guiding_center = max(1000, int(maxtime/1.0e-8))
timesteps_lorentz = 1000
nparticles = number_of_cores * number_of_particles_per_core  # High fidelity particle count (e.g., 24)
n_segments = 80
coil_current = 7e6
change_currents = False
model = 'Lorentz'  # 'Guiding Center' or 'Lorentz'
method = 'least_squares'  # Not used in this snippet
max_function_evaluations = 5
max_iterations_BFGS = 1000
max_function_evaluations_BFGS = 400
max_function_evaluations_BOBYQA = 550
tolerance_to_terminace_optimization = 1e-6
min_val = -11  # minimum coil dof value
max_val = 15   # maximum coil dof value
##### Input parameters stop here

particles = Particles(nparticles)

candidate_particle_counts = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([coil_current] * n_curves))
timesteps = timesteps_lorentz if model == 'Lorentz' else timesteps_guiding_center

x0, y0, z0, vpar0, vperp0 = stel.initial_conditions(particles, R, r_init, model='Guiding Center')
v0 = jnp.zeros((3, nparticles))
for i in range(nparticles):
    B0 = B(jnp.array([x0[i], y0[i], z0[i]]), curve_segments=stel.gamma(n_segments), currents=stel.currents)
    b0 = B0 / jnp.linalg.norm(B0)
    perp_vector_1 = jnp.array([0, b0[2], -b0[1]])
    perp_vector_1_normalized = perp_vector_1 / jnp.linalg.norm(perp_vector_1)
    perp_vector_2 = jnp.cross(b0, perp_vector_1_normalized)
    v0 = v0.at[:, i].set(vpar0[i] * b0 + vperp0[i] * (perp_vector_1_normalized / jnp.sqrt(2) + perp_vector_2 / jnp.sqrt(2)))
normB0 = jnp.apply_along_axis(B_norm, 0, jnp.array([x0, y0, z0]), stel.gamma(n_segments), stel.currents)
μ = particles.mass * vperp0**2 / (2 * normB0)
start = time()
loss_value = loss(stel.dofs, stel.dofs_currents, stel, particles, R, r_init,
                    jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]),
                    maxtime, timesteps, n_segments, model=model)
print(f"Loss function initial value: {loss_value:.8f}")
end = time()
print(f"Took: {end - start:.2f} seconds")

initial_values = (jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]])
                  if model == 'Lorentz' else
                  jnp.array([x0, y0, z0, vpar0, vperp0]))

loss_partial = partial(loss, old_coils=stel, particles=particles, R=R, r_init=r_init,
                       initial_values=initial_values, maxtime=maxtime, timesteps=timesteps,
                       n_segments=n_segments, model=model)

len_dofs = len(jnp.ravel(stel.dofs))
if change_currents:
    all_dofs = jnp.concatenate((jnp.ravel(stel.dofs), jnp.ravel(stel.currents)[:n_curves]))
else:
    all_dofs = jnp.ravel(stel.dofs)

print(f"Number of dofs: {len(all_dofs)}")

results = []

for candidate in candidate_particle_counts:
    print(f"\n=== Running simulation for low fidelity with {candidate} particles ===")
    reduced_particles = Particles(candidate)
    
    iteration = 0
    bias = 0
    first_iter = True
    objective_value = []
    flag_value = []
    
    if change_currents:
        all_dofs = jnp.concatenate((jnp.ravel(stel.dofs), jnp.ravel(stel.currents)[:n_curves]))
    else:
        all_dofs = jnp.ravel(stel.dofs)
    
    start_optimization_time = time()
    res = minimize(multifidelity_loss, x0=all_dofs, method="L-BFGS-B",
                options={"maxiter": 30, "disp": False},
                   callback=callback)
    end_optimization_time = time()
    optimization_duration = end_optimization_time - start_optimization_time
    
    x_final = jnp.array(res.x)
    if change_currents:
        dofs_final = jnp.reshape(x_final[:len_dofs], shape=stel.dofs.shape)
    else:
        dofs_final = jnp.reshape(x_final, shape=stel.dofs.shape)
    
    final_high_fidelity_loss = high_fidelity_loss(dofs_final, stel.dofs_currents, stel, particles,
                                                  R, r_init, initial_values, maxtime, timesteps,
                                                  n_segments, model)
    
    results.append({
        "candidate": candidate,
        "final_high_fidelity_loss": float(final_high_fidelity_loss),
        "final_dofs": x_final.tolist(),
        "duration": optimization_duration
    })
    print(f"Completed simulation for candidate {candidate} particles: Final High Fidelity Loss = {final_high_fidelity_loss:.8f}, Duration = {optimization_duration:.2f} seconds\n")

sorted_results = sorted(results, key=lambda r: r["final_high_fidelity_loss"])
print("=== Sorted Results by High Fidelity Loss ===")
for r in sorted_results:
    print(f"Candidate low fidelity particles: {r['candidate']}, High Fidelity Loss: {r['final_high_fidelity_loss']:.8f}, Duration: {r['duration']:.2f} seconds")

with open("results.txt", "w") as f:
    f.write("Sorted Results by High Fidelity Loss:\n")
    for r in sorted_results:
        f.write(f"Candidate low fidelity particles: {r['candidate']}, High Fidelity Loss: {r['final_high_fidelity_loss']:.8f}, Duration: {r['duration']:.2f} seconds\n")

plt.plot(objective_value, label='High Fidelity Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
