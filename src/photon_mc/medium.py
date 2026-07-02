"""Slab medium geometry and optical properties."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Medium:
    """Optical properties and geometry of a slab medium (0 <= z <= thickness)."""
    mu_a: float           # absorption coefficient [1/mm]
    mu_s: float           # scattering coefficient [1/mm]
    n: float              # refractive index of the medium (dimensionless)
    n_outside: float      # refractive index of the surrounding material outside both faces
    thickness: float      # slab thickness [mm]; the medium occupies 0 <= z <= thickness

    @property
    def l_scat(self) -> float:
        """Scattering mean free path (mean step length) [mm]."""
        return 1.0 / self.mu_s

    @property
    def l_abs(self) -> float:
        """Absorption length [mm]."""
        return 1.0 / self.mu_a
