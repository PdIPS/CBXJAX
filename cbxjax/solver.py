"""
cbxjax/solver.py
 
Abstract base class for consensus-based solvers and the run loop.
"""
 
from abc import ABC, abstractmethod
from typing import Callable, Tuple
import jax
from jax import lax
 
from .base import CBXState
 
 
def apply_fns(fns, state: CBXState) -> CBXState:
    for fn in fns:
        state = fn(state)
    return state
 
 
class CBXDynamic(ABC):
    """
    Abstract base class for consensus-based solvers.
 
    Subclasses must implement:
        - kernel(state) -> state   the dynamic update
        - init_state(f, x0, key, **kwargs) -> CBXState
 
    Subclasses may optionally set:
        - pre_fns  : list of state -> state, applied before kernel
        - post_fns : list of state -> state, applied after kernel
    """
 
    pre_fns  : tuple = ()
    post_fns : tuple = ()
 
    @abstractmethod
    def kernel(self, state: CBXState) -> CBXState:
        """
        The core dynamic update. Takes a state and returns a new state.
        Must update at minimum: x, energy, consensus.
        """
        ...
 
    @abstractmethod
    def init_state(
        self,
        f   : Callable,
        x0  : jax.Array,
        key : jax.Array,
    ) -> CBXState:
        """
        Build the initial CBXState for this solver.
        """
        ...
 
    def step(self, state: CBXState) -> CBXState:
        """
        Full step: pre_fns -> kernel -> post_fns.
        This is what lax.scan iterates over.
        """
        state = apply_fns(self.pre_fns, state)
        state = self.kernel(state)
        state = apply_fns(self.post_fns, state)
        return state
 
    def run(self, x0, key, max_it=1000, track_fns=()):
        state = self.init_state(x0, key)

        def scan_step(state: CBXState, _):
            state   = self.step(state)
            tracked = tuple(fn(state) for fn in track_fns)
            return state, tracked
        return lax.scan(scan_step, state, None, length=max_it)