"""Registry of selectable scattering phase functions.

Each phase function lives in its own module in this package and exposes a
class with a `sample_cos_theta(xi: torch.Tensor) -> torch.Tensor` method that
inverts its CDF given a batch of U(0, 1) random variates (already drawn by
the caller's RNG) and returns the corresponding polar scattering cosines.

A phase function may optionally define `mueller_matrix(cos_theta) -> (N,4,4)
torch.Tensor`, used by the simulation's polarization tracking instead of the
generic Rayleigh approximation in `phase_functions.mueller` (see `Mie`,
which derives both the sampled angle and the Mueller matrix from the same
physical particle model). It may also define `to(device, dtype)` if it
holds precomputed tensors that need to move onto the simulation's device.

To add a new phase function: drop a module here with such a class and
register it in PHASE_FUNCTIONS below. Select it from script_v0.py via the
PHASE_FUNCTION / PHASE_FUNCTION_PARAMS parameters at the top of that file.
"""

from __future__ import annotations

from .henyey_greenstein import HenyeyGreenstein
from .isotropic import Isotropic
from .mie import Mie

PHASE_FUNCTIONS = {
    "henyey_greenstein": HenyeyGreenstein,
    "isotropic": Isotropic,
    "mie": Mie,
}


def get_phase_function(name: str, **params):
    """Construct the selected phase function by name with the given params."""
    try:
        cls = PHASE_FUNCTIONS[name]
    except KeyError:
        raise ValueError(f"Unknown phase function {name!r}. Available: {sorted(PHASE_FUNCTIONS)}") from None
    return cls(**params)
