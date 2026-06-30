"""Tissue layer definition with optical properties."""

from dataclasses import dataclass


@dataclass
class TissueLayer:
    """A homogeneous tissue slab.

    Attributes
    ----------
    mu_a  : absorption coefficient (1/mm)
    mu_s  : scattering coefficient (1/mm)
    g     : Henyey-Greenstein anisotropy factor [-1, 1]
    n     : refractive index
    depth : physical thickness (mm); use float('inf') for semi-infinite
    """

    mu_a: float
    mu_s: float
    g: float
    n: float
    depth: float = float("inf")

    def __post_init__(self) -> None:
        if self.mu_a < 0 or self.mu_s < 0:
            raise ValueError("Absorption and scattering coefficients must be non-negative.")
        if not (-1.0 <= self.g <= 1.0):
            raise ValueError("Anisotropy g must be in [-1, 1].")
        if self.n <= 0:
            raise ValueError("Refractive index must be positive.")

    @property
    def mu_t(self) -> float:
        """Total attenuation coefficient."""
        return self.mu_a + self.mu_s

    @property
    def albedo(self) -> float:
        """Single-scattering albedo."""
        return self.mu_s / self.mu_t if self.mu_t > 0 else 0.0
