"""Polarization-resolved Monte Carlo photon transport in a slab scattering/absorbing medium.

See CLAUDE.md for the full physical model and `python -m photon_mc` to run
the configured simulation (see `photon_mc.params`).
"""

from __future__ import annotations

from .medium import Medium
from .phase_functions import get_phase_function
from .simulation import MonteCarlo

__all__ = ["Medium", "MonteCarlo", "get_phase_function"]
