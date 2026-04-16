from typing import Callable, List, Union
import jax
import jax.numpy as jnp
 
from .base import CBXState, compute_consensus, update_best
from .solver import CBXDynamic
 
 
# ---------------------------------------------------------------------------
# Shared init helper
# ---------------------------------------------------------------------------
 
def _init_state(
    f     : Callable,
    x0    : jax.Array,
    key   : jax.Array,
    alpha : float,
    sigma : float,
    dt    : float,
) -> CBXState:
    x0 = jnp.asarray(x0, dtype=float)
    if x0.ndim == 2:
        x0 = x0[None]                              # (N, d) -> (1, N, d)
 
    M, N, d = x0.shape
    energy    = f(x0)                        # (M, N)
    consensus = jnp.mean(x0, axis=-2, keepdims=True)
 
    best_idx    = jnp.argmin(energy, axis=-1)
    best_energy = energy[jnp.arange(M), best_idx]
    best_part   = x0[jnp.arange(M), best_idx, :]
 
    key, subkey = jax.random.split(key)
 
    return CBXState(
        x             = x0,
        energy        = energy,
        consensus     = consensus,
        best_particle = best_part,
        best_energy   = best_energy,
        active        = jnp.ones(M, dtype=bool),
        it            = jnp.zeros((), dtype=int),
        t             = jnp.zeros(()),
        alpha         = jnp.full((M, 1), alpha),
        sigma         = jnp.full((M, 1), sigma),
        dt            = jnp.asarray(dt),
        key           = subkey,
    )

# ---------------------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------------------
 
def isotropic_noise(key, drift):
    scale = jnp.linalg.norm(drift, axis=-1, keepdims=True)
    return scale * jax.random.normal(key, shape=drift.shape)
 
 
def anisotropic_noise(key, drift):
    return drift * jax.random.normal(key, shape=drift.shape)
 
 
def _resolve_noise(noise):
    if noise == 'isotropic':
        return isotropic_noise
    elif noise == 'anisotropic':
        return anisotropic_noise
    elif callable(noise):
        return noise
    raise ValueError(f"Unknown noise model: {noise!r}")

# ---------------------------------------------------------------------------
# CBO
# ---------------------------------------------------------------------------
@jax.jit
def cbo_update(drift, dt, sigma, noise):
    return -dt * drift + jnp.sqrt(dt) * sigma[:, :, None] * noise

 
class CBO(CBXDynamic):
    def __init__(
        self,
        f        : Callable,
        alpha    : float = 10.0,
        sigma    : float = 1.0,
        dt       : float = 0.01,
        noise    : Union[str, Callable] = 'anisotropic',
        pre_fns  : List[Callable] = (),
        post_fns : List[Callable] = (),
    ):
        self.f        = f
        self.alpha    = alpha
        self.sigma    = sigma
        self.dt       = dt
        self.noise_fn = _resolve_noise(noise)
        self.pre_fns  = tuple(pre_fns)
        self.post_fns = tuple(post_fns)
 
    def init_state(self, x0, key) -> CBXState:
        return _init_state(self.f, x0, key, self.alpha, self.sigma, self.dt)
 
    def kernel(self, state: CBXState) -> CBXState:
        key, subkey = jax.random.split(state.key)
 
        consensus = compute_consensus(state.x, state.energy, state.alpha)
        drift     = state.x - consensus                                    # (M, N, d)
        noise     = self.noise_fn(subkey, drift)
 
        x_new = state.x + cbo_update(drift, state.dt, state.sigma, noise)
        x_new = jnp.where(state.active[:, None, None], x_new, state.x)
 
        energy_new                 = self.f(x_new)
        best_particle, best_energy = update_best(state, x_new, energy_new)
 
        return state._replace(
            x             = x_new,
            energy        = energy_new,
            consensus     = consensus,
            best_particle = best_particle,
            best_energy   = best_energy,
            it            = state.it + 1,
            t             = state.t + state.dt,
            key           = key,
        )
 