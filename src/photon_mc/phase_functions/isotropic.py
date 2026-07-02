"""Isotropic scattering phase function (uniform over the sphere)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Isotropic:
    """Scatters with equal probability in every direction (no parameters)."""
    name = "isotropic"

    def sample_cos_theta(self, xi: torch.Tensor) -> torch.Tensor:
        return 2.0 * xi - 1.0
