import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# 2D functions
# ---------------------------------------------------------------------------

@jax.jit
def three_hump_camel(x):
    """Global minimum at (0, 0) with f=0."""
    return (2*x[...,0]**2
            - 1.05*x[...,0]**4
            + x[...,0]**6 / 6
            + x[...,0]*x[...,1]
            + x[...,1]**2)


@jax.jit
def McCormick(x):
    """Global minimum at (-0.54719, -1.54719) with f=-1.9133."""
    return (jnp.sin(x[...,0] + x[...,1])
            + (x[...,0] - x[...,1])**2
            - 1.5*x[...,0]
            + 2.5*x[...,1]
            + 1)


@jax.jit
def Himmelblau(x):
    """Four global minima with f=0."""
    return ((x[...,0]**2 + x[...,1] - 11)**2
            + (x[...,0] + x[...,1]**2 - 7)**2)


@jax.jit
def snowflake(x, alpha=0.5):
    """
    Six global minima. Introduced for PolarCBO benchmarking.
    (Bungert, Roith, Wacker 2022)
    """
    z   = alpha * x
    r   = jnp.linalg.norm(z, axis=-1)
    phi = jnp.arctan2(z[...,1], z[...,0])

    res = jnp.ones_like(r)
    for psi in [0., jnp.pi/3, 2*jnp.pi/3]:
        g   = r**8 - r**4 + jnp.abs(jnp.cos(phi + psi))**0.5 * r**0.3
        res = jnp.minimum(res, g)

    return jnp.minimum(res, 0.8)


@jax.jit
def eggholder(x):
    """Global minimum at (512, 404.2319) with f=-959.6407."""
    return (-(x[...,1] + 47) * jnp.sin(jnp.sqrt(jnp.abs(x[...,1] + x[...,0]/2 + 47)))
            - x[...,0] * jnp.sin(jnp.sqrt(jnp.abs(x[...,0] - x[...,1] - 47))))


# ---------------------------------------------------------------------------
# d-dimensional functions
# ---------------------------------------------------------------------------

@jax.jit
def Rosenbrock(x, a=1., b=100.):
    """
    Global minimum at (a, a^2) with f=0.
    Standard 2D version: sum over consecutive pairs.
    """
    return jnp.sum(
        b * (x[..., 1:] - x[..., :-1]**2)**2 + (a - x[..., :-1])**2,
        axis=-1
    )


@jax.jit
def Rastrigin(x, b=0., c=0., A=10.):
    """Global minimum at (b,...,b) with f=c."""
    z = x - b
    return A * x.shape[-1] + jnp.sum(z**2 - A * jnp.cos(2*jnp.pi*z), axis=-1) + c


@jax.jit
def Ackley(x, A=20., b=0.2, c=2*jnp.pi):
    """Global minimum at origin with f=0."""
    d    = x.shape[-1]
    arg1 = -b * jnp.sqrt(jnp.sum(x**2, axis=-1) / d)
    arg2 = jnp.sum(jnp.cos(c * x), axis=-1) / d
    return -A * jnp.exp(arg1) - jnp.exp(arg2) + A + jnp.e


@jax.jit
def Michalewicz(x, m=10):
    """
    Global minimum depends on d; for d=2 at (2.2029, 1.5708) with f≈-1.8013.
    """
    d   = x.shape[-1]
    idx = jnp.arange(1, d + 1)
    return -jnp.sum(jnp.sin(x) * jnp.sin(idx * x**2 / jnp.pi)**(2*m), axis=-1)


@jax.jit
def quadratic(x):
    """f(x) = ||x||^2. Trivial test case, minimum at origin."""
    return jnp.sum(x**2, axis=-1)


# ---------------------------------------------------------------------------
# Parametrized constructors  (return a jitted function)
# Useful when parameters are not known at import time.
# ---------------------------------------------------------------------------

def make_Rastrigin(b=0., c=0., A=10.):
    return jax.jit(partial(Rastrigin, b=b, c=c, A=A))

def make_Ackley(A=20., b=0.2, c=2*jnp.pi):
    return jax.jit(partial(Ackley, A=A, b=b, c=c))

def make_Rosenbrock(a=1., b=100.):
    return jax.jit(partial(Rosenbrock, a=a, b=b))

def make_snowflake(alpha=0.5):
    return jax.jit(partial(snowflake, alpha=alpha))


# ---------------------------------------------------------------------------
# Known minima  (for test assertions)
# ---------------------------------------------------------------------------

MINIMA = {
    'quadratic'       : [(0., 0.)],
    'three_hump_camel': [(0., 0.)],
    'mccormick'       : [(-0.54719, -1.54719)],
    'himmelblau'      : [(3., 2.), (-2.805118, 3.131312),
                         (-3.779310, -3.283186), (3.584428, -1.848126)],
    'rosenbrock'      : [(1., 1.)],
    'eggholder'       : [(512., 404.2319)],
    'michalewicz_2d'  : [(2.2029, 1.5708)],
}