"""Henyey-Greenstein scattering phase function."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HenyeyGreenstein:
    """Henyey-Greenstein phase function, parameterized by anisotropy g.

    g = 0 is isotropic; g -> 1 is strongly forward-peaked (typical of
    tissue-like media); g -> -1 is strongly backward-peaked.
    """
    name = "henyey_greenstein"
    g: float = 0.0   # anisotropy factor, -1 < g < 1

    def sample_cos_theta(self, xi: torch.Tensor) -> torch.Tensor:
        """Invert the Henyey-Greenstein CDF to draw polar scattering cosines."""
        g = self.g
        if abs(g) < 1e-8:
            return 2.0 * xi - 1.0  # isotropic limit, avoids division by g
        s = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi)
        return (1.0 + g * g - s * s) / (2.0 * g)
