from ESSOS import Coils, Particles, loss, CreateEquallySpacedCurves

import os
import jax
import sys
import pybobyqa  # type: ignore
from jax import jit
from time import time
import jax.numpy as jnp
from matplotlib import cm
from functools import partial
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from scipy.optimize import least_squares, minimize
import cma

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
first_iter = True

def multifidelity_loss(x):
    global bias, first_iter
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    if first_iter:
        high_fidelity_value = high_fidelity_loss(dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        low_fidelity_value = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        bias = high_fidelity_value - low_fidelity_value
        print(f"HighFidelity Step: HF Loss {high_fidelity_value} --- LF Loss {low_fidelity_value} --- Bias {bias}")
        first_iter = False
    else:
        low_fidelity_value = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    
    return low_fidelity_value

#### INPUT PARAMETERS START HERE ####
number_of_cores = 3
number_of_particles_per_core = 8
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_cores}'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])

sys.path.insert(1, os.path.dirname(os.getcwd()))
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, loss
from MagneticField import B, B_norm

n_curves = 2
nfp = 4
order = 2
r = 3
A = 2
R = A * r
r_init = r / 4
maxtime = 3.0e-6
timesteps_lorentz = 1000
nparticles = number_of_cores * number_of_particles_per_core
n_segments = 80
coil_current = 7e6
change_currents = False
model = 'Lorentz'
min_val = -11
max_val = 15
particles = Particles(nparticles)
reduced_particles = Particles(max(1, nparticles // 3))
curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([coil_current] * n_curves))
timesteps = timesteps_lorentz

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
μ = particles.mass * vperp0 ** 2 / (2 * normB0)
initial_values = jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]])

len_dofs = len(jnp.ravel(stel.dofs))
if change_currents:
    all_dofs = jnp.concatenate((jnp.ravel(stel.dofs), jnp.ravel(stel.currents)[:n_curves]))
else:
    all_dofs = jnp.ravel(stel.dofs)

bounds = [min_val, max_val]
sigma = 0.5
x0 = all_dofs.tolist()
es = cma.CMAEvolutionStrategy(x0, sigma, {'bounds': bounds, 'popsize': 20, 'maxiter': 10})

objective_value = []

while not es.stop():
    solutions = es.ask()
    evaluations = [float(multifidelity_loss(jnp.array(sol))) for sol in solutions]
    objective_value.extend(evaluations)
    es.tell(solutions, evaluations)
    es.disp()


x_opt = es.result.xbest
stel.dofs = jnp.reshape(jnp.array(x_opt[:len_dofs]), stel.dofs.shape)

end_optimization_time = time()
print("Optimization completed with CMA-ES.")
print(f"Optimized parameters: {x_opt}")

plt.plot(objective_value, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
