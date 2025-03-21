from ESSOS import Coils, Particles, loss, CreateEquallySpacedCurves

import os
import jax
import sys
import pybobyqa # type: ignore
from jax import jit, grad
from time import time
import jax.numpy as jnp
from matplotlib import cm
from functools import partial
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from scipy.optimize import minimize
from functools import partial
import time
import jax.numpy as np
import math

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

def create_grad_f_hi(dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model):
    def hf_loss_wrapped(dofs):
        return high_fidelity_loss(dofs, dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    grad_hf_loss = grad(hf_loss_wrapped)
    return grad_hf_loss

def hf_loss_func(z, dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model):
    return high_fidelity_loss(z, dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)

def lf_loss_func(z, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model):
    return low_fidelity_loss(z, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model)

def f_lo_adjusted(x, z, low_fidelity_loss_z, high_fidelity_loss_z, grad_high_z, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model):
    return (low_fidelity_loss(x, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        + (high_fidelity_loss_z - low_fidelity_loss_z)
        + grad_high_z @ (x - z))

def step_loss(s, z, low_fidelity_loss_z, high_fidelity_loss_z, grad_high_z, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model):
    t_start = time.time()
    loss_value = f_lo_adjusted(z + s, z, low_fidelity_loss_z, high_fidelity_loss_z, grad_high_z, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    print(f"    Function evaluation | Time: {time.time() - t_start:.6f}s")
    return loss_value

def grad_step(s, z, low_fidelity_loss_z, high_fidelity_loss_z, grad_high_z, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model):
    t_start = time.time()
    def step_loss_wrapped(s_):
        return step_loss(s_, z, low_fidelity_loss_z, high_fidelity_loss_z, grad_high_z, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    grad_value = grad(step_loss_wrapped)(s)
    print(f"    Gradient evaluation | Time: {time.time() - t_start:.6f}s")
    return grad_value

#### INPUT PARAMETERS START HERE - NUMBER OF PARTICLES ####
number_of_cores = 3
number_of_particles_per_core = 9
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_cores}'
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])
sys.path.insert(1, os.path.dirname(os.getcwd()))
from ESSOS import CreateEquallySpacedCurves, Coils, Particles, set_axes_equal, loss
from MagneticField import B, B_norm
n_curves = 2
nfp = 4
order = 2
r = 3
A = 2  # Aspect ratio
R = A * r
r_init = r / 4
maxtime = 3.0e-6
timesteps_guiding_center = max(1000, int(maxtime / 1.0e-8))
timesteps_lorentz = 1000
nparticles = number_of_cores * number_of_particles_per_core
n_segments = 80
coil_current = 7e6
change_currents = False
model = 'Lorentz'
particles = Particles(nparticles)
reduced_particles = Particles(max(1, nparticles // 3))
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

initial_values = jnp.array([x0, y0, z0, v0[0], v0[1], v0[2]]) if model == 'Lorentz' else jnp.array([x0, y0, z0, vpar0, vperp0])

loss_partial = partial(
    loss,
    old_coils=stel,
    particles=particles,
    R=R,
    r_init=r_init,
    initial_values=initial_values,
    maxtime=maxtime,
    timesteps=timesteps,
    n_segments=n_segments,
    model=model
)
len_dofs = len(jnp.ravel(stel.dofs))

if change_currents:
    all_dofs = jnp.concatenate((jnp.ravel(stel.dofs), jnp.ravel(stel.currents)[:n_curves]))
else:
    all_dofs = jnp.ravel(stel.dofs)

def trust_region_optimization(z0, dofs_currents, coils, particles, R, r_init, initial_values, 
                              maxtime, timesteps, n_segments, model, grad_f_hi,
                              delta_max=0.2, gamma1=0.15, gamma2=0.3,
                              trust_region_sensitivity=1.0, max_iter=40, tol=1e-4, loss_req=0.125):
    z = z0
    iteration = 0
    delta = delta_max * 0.4
    history = [z0.copy()]
    obj_vals = []
    total_iterations = 0
    delta_vals = [delta] 
    
    print("\nStarting Trust-Region Optimization...")
    
    while iteration < max_iter:
        print(f"\nIteration {iteration} | Delta = {delta:.6f}")

        t0 = time.time()
        high_fidelity_loss_z = high_fidelity_loss(z, dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        print(f"  High-fidelity loss | Time: {time.time() - t0:.6f}s | Loss: {high_fidelity_loss_z:.6f}")
        obj_vals.append(high_fidelity_loss_z)

        t1 = time.time()
        low_fidelity_loss_z = low_fidelity_loss(z, dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        print(f"  Low-fidelity loss  | Time: {time.time() - t1:.6f}s | Loss: {low_fidelity_loss_z:.6f}")

        t2 = time.time()
        grad_high_z = grad_f_hi(z)
        print(f"  Gradient computed  | Time: {time.time() - t2:.6f}s")

        step_loss_partial = partial(step_loss, z=z, low_fidelity_loss_z=low_fidelity_loss_z,
                                    high_fidelity_loss_z=high_fidelity_loss_z, grad_high_z=grad_high_z,
                                    dofs_currents=dofs_currents, coils=coils, R=R, r_init=r_init,
                                    initial_values=initial_values, maxtime=maxtime, timesteps=timesteps,
                                    n_segments=n_segments, model=model)

        grad_step_partial = partial(grad_step, z=z, low_fidelity_loss_z=low_fidelity_loss_z,
                                    high_fidelity_loss_z=high_fidelity_loss_z, grad_high_z=grad_high_z,
                                    dofs_currents=dofs_currents, coils=coils, R=R, r_init=r_init,
                                    initial_values=initial_values, maxtime=maxtime, timesteps=timesteps,
                                    n_segments=n_segments, model=model)

        def callback(x):
            nonlocal total_iterations
            total_iterations += 1

        print("  Running L-BFGS-B...")
        t3 = time.time()
        res = minimize(
            step_loss_partial,
            np.zeros_like(z),
            jac=grad_step_partial,
            bounds=[(-delta, delta) for _ in z],
            method="L-BFGS-B",
            options={"maxiter": 30, "disp": False, "ftol": 1e-4, "gtol": 1e-4},
            callback=callback
        )
        print(f"  L-BFGS-B completed | Time: {time.time() - t3:.6f}s")
        print(f"  Total iterations so far: {total_iterations}")

        s = res.x
        t4 = time.time()
        f_hi_z_after_s = high_fidelity_loss(z + s, dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
        print(f"  High-fidelity loss (new point) | Time: {time.time() - t4:.6f}s | Loss: {f_hi_z_after_s:.6f}")

        actual_reduction = high_fidelity_loss_z - f_hi_z_after_s
        predicted_reduction = high_fidelity_loss_z - f_lo_adjusted(
            z + s, z, low_fidelity_loss_z, high_fidelity_loss_z, grad_high_z,
            dofs_currents, coils, R, r_init, initial_values, maxtime, timesteps, n_segments, model
        )
        gamma = actual_reduction / predicted_reduction if predicted_reduction > 0 else 0

        print(f"  Actual Reduction: {actual_reduction:.6f} | Predicted Reduction: {predicted_reduction:.6f} | Gamma: {gamma:.4f}")

        if gamma < 0:
            delta = min(delta * (1 + trust_region_sensitivity * abs(gamma)), delta_max) #expand to escpace
        elif gamma < gamma1:
            multiplier = math.exp(-trust_region_sensitivity * (gamma1 - gamma)) #contract
            delta = max(delta * multiplier, tol)
        elif gamma < gamma2:
            delta = delta
        else:
            delta = min(delta * (1 + trust_region_sensitivity * (gamma - gamma2)), delta_max) #expand

        delta_vals.append(delta) 

        print(f"  Updated Delta: {delta:.6f}")

        if high_fidelity_loss_z < loss_req:
            print("  Loss requirement met. Terminating optimization.")
            break

        #if actual_reduction > 0:
        z = z + s
        history.append(z.copy())

        iteration += 1

    return z, obj_vals, history, delta_vals


grad_f_hi = create_grad_f_hi(
    dofs_currents=stel.dofs_currents,
    coils=stel,
    particles=particles,
    R=R,
    r_init=r_init,
    initial_values=initial_values,
    maxtime=maxtime,
    timesteps=timesteps,
    n_segments=n_segments,
    model=model
)
t_trust = time.time()
z0 = jnp.ravel(stel.dofs)
z_opt, obj_vals, history, delta_vals = trust_region_optimization(
    z0=z0,
    dofs_currents=stel.dofs_currents,
    coils=stel,
    particles=particles,
    R=R,
    r_init=r_init,
    initial_values=initial_values,
    maxtime=maxtime,
    timesteps=timesteps,
    n_segments=n_segments,
    model=model,
    grad_f_hi=grad_f_hi
)
print(f"Multifidelity Trust Time: {time.time() - t_trust:.6f}s")

if change_currents:
    all_dofs = jnp.concatenate((jnp.ravel(stel.dofs), jnp.ravel(stel.currents)[:n_curves]))
else:
    all_dofs = jnp.ravel(stel.dofs)

iteration = 0
obj_vals_low = []

def low_loss(x):
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    low_fidelity_value = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    return low_fidelity_value

def callback_low(x, res=None):
    global flag_value,iteration, obj_vals_low

    print("callback")

    print(f"Iteration {iteration}:")
    print("Objective Function: {}".format(low_loss(x)))

    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    hf_val = high_fidelity_loss(dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    obj_vals_low += [hf_val]

    low_fidelity_value = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    print(f"HighFidelity Step: HF Loss {hf_val} --- LF Loss {low_fidelity_value}")

    iteration += 1

res = minimize(low_loss, x0=all_dofs, method="L-BFGS-B", options={ 'maxiter': 40, "disp": False, "ftol": 1e-4, "gtol": 1e-4}, callback = callback_low, tol=1e-8)

iteration = 0
def low_loss_time(x):
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    low_fidelity_value = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    return low_fidelity_value

def callback_low_time(x, res=None):
    global flag_value,iteration, obj_vals_low

    print("callback")

    print(f"Iteration {iteration}:")
    print("Objective Function: {}".format(low_loss(x)))

    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    low_fidelity_value = low_fidelity_loss(dofs, stel.dofs_currents, stel, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    print(f"LF Loss {low_fidelity_value}")

    iteration += 1

t_low = time.time()
res = minimize(low_loss_time, x0=all_dofs, method="L-BFGS-B", options={ 'maxiter': 40, "disp": False, "ftol": 1e-4, "gtol": 1e-4}, callback = callback_low_time, tol=1e-8)
print(f"Low Time: {time.time() - t_low:.6f}s")


if change_currents:
    all_dofs = jnp.concatenate((jnp.ravel(stel.dofs), jnp.ravel(stel.currents)[:n_curves]))
else:
    all_dofs = jnp.ravel(stel.dofs)

iteration = 0
obj_vals_high = []

def high_loss(x):
    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    high_fidelity_value = high_fidelity_loss(dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    return high_fidelity_value

def callback_high(x, res=None):
    global flag_value, iteration, bias, obj_vals_high

    print("callback")

    print(f"Iteration {iteration}:")
    print("Objective Function: {}".format(high_loss(x)))

    dofs, currents = jax.lax.cond(
        change_currents,
        lambda _: (jnp.reshape(x[:len_dofs], shape=stel.dofs.shape), x[-n_curves:]),
        lambda _: (jnp.reshape(x, shape=stel.dofs.shape), stel.currents[:n_curves]),
        operand=None
    )
    hf_val = high_fidelity_loss(dofs, stel.dofs_currents, stel, particles, R, r_init, initial_values, maxtime, timesteps, n_segments, model)
    obj_vals_high += [hf_val]
    print(f"HighFidelity Step: HF Loss {hf_val}")

    iteration += 1

t_high = time.time()
res = minimize(high_loss, x0=all_dofs, method="L-BFGS-B", options={ 'maxiter': 40, "disp": False, "ftol": 1e-4, "gtol": 1e-4}, callback = callback_high, tol=1e-8)
print(f"High Time: {time.time() - t_high:.6f}s")


plt.plot(obj_vals, label="Multifidelity Trust Region Optimization")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Optimization Convergence Comparison")
plt.legend()
plt.grid()

plt.plot(obj_vals_low, label="Low Fidelity Optimization", color="red")

plt.plot(obj_vals_high, label="High Fidelity Optimization", color="green")

plt.show()
