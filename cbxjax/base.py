
from typing import Callable, NamedTuple, Tuple
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import lax
 
 
# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
 
class CBXState(NamedTuple):
    """
    Immutable state pytree for consensus-based dynamics.
 
    Shapes assume M independent runs, N particles, d-dimensional space.
 
    Fields
    ------
    x               : (M, N, d)   particle positions
    energy          : (M, N)      last evaluated f(x)
    best_particle   : (M, d)      best particle found so far (across all steps)
    best_energy     : (M,)        corresponding best energy
    active          : (M,)  bool  False once a run's termination criterion is met
    it              : ()    int   current iteration index
    t               : ()    float current time  t = it * dt
    alpha           : (M, 1)      softmax inverse-temperature (schedulable)
    sigma           : (M, 1)      noise scale (schedulable)
    dt              : ()          time-step size
    key             : (2,)  uint32  JAX PRNG key
    """
    x             : jax.Array          # (M, N, d)
    energy        : jax.Array          # (M, N)
    consensus     : jax.Array          # (M, 1, d)
    best_particle : jax.Array          # (M, d)
    best_energy   : jax.Array          # (M,)
    active        : jax.Array          # (M,)  bool
    it            : jax.Array          # ()    int
    t             : jax.Array          # ()    float
    alpha         : jax.Array          # (M, 1)
    sigma         : jax.Array          # (M, 1)
    dt            : jax.Array          # ()
    key           : jax.Array          # (2,)  uint32

# ---------------------------------------------------------------------------
# Consensus computation  (shared across all variants)
# ---------------------------------------------------------------------------

@jax.jit
def compute_consensus(
    x     : jax.Array,   # (M, N, d)
    energy: jax.Array,   # (M, N)
    alpha : jax.Array,   # (M, 1)
) -> jax.Array:          # (M, 1, d)
    weights = - alpha * energy
    coeffs = jnp.exp(weights - jsp.logsumexp(weights, axis=-1, keepdims=True))
    #return (x * coeffs).sum(axis=1, keepdims=True)
    return jnp.einsum("ij,ijk->ik", coeffs, x)[:, None, :]


# ------------------------------------------------------------------
# Update best particle
# ------------------------------------------------------------------

@jax.jit
def update_best(state, x_new, energy_new):
    cur_best_idx    = jnp.argmin(energy_new, axis=-1)              # (M,)
    cur_best_energy = energy_new[jnp.arange(energy_new.shape[0]), cur_best_idx]
    cur_best_part   = x_new[jnp.arange(x_new.shape[0]), cur_best_idx, :]

    improved      = cur_best_energy < state.best_energy            # (M,)
    best_energy   = jnp.where(improved, cur_best_energy, state.best_energy)
    best_particle = jnp.where(improved[:, None], cur_best_part, state.best_particle)
    return best_particle, best_energy