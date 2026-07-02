"""Exact Mie-theory phase function and Mueller matrix, via miepython.

Unlike the other phase functions in this package, `Mie` also defines its own
`mueller_matrix()`: the scattering angle and the polarization evolution are
both derived from the same physical model (a sphere of refractive index `m`
and physical size `radius`/`wavelength`), instead of pairing an arbitrary
direction-sampling phase function (Henyey-Greenstein/isotropic) with the
unrelated Rayleigh approximation in `phase_functions.mueller`.

Mie scattering has no closed-form CDF, so S11(cos_theta) and the full
Mueller matrix are tabulated once on a uniform grid in cos(theta) at
construction time (a one-time call into miepython, on CPU/numpy), then
inverted/interpolated with fast batched linear interpolation at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import miepython
import numpy as np
import torch


@dataclass
class Mie:
    """Mie-theory phase function + Mueller matrix for a sphere.

    m: complex refractive index of the sphere relative to the medium (dimensionless ratio)
    radius: physical particle radius [um]
    wavelength: wavelength of light in the medium [um]
    n_mu: resolution of the cos(theta) table backing CDF inversion and the
        Mueller-matrix interpolation (uniform grid on [-1, 1]); a sharply
        forward-peaked phase function (large size parameter) needs more
        points to resolve the forward lobe.
    """
    name = "mie"
    m: complex
    radius: float
    wavelength: float
    n_mu: int = 4000
    x: float = field(init=False)  # dimensionless Mie size parameter, derived below

    def __post_init__(self):
        self.x = 2.0 * math.pi * self.radius / self.wavelength

        mu = np.linspace(-1.0, 1.0, self.n_mu)
        p = miepython.phase_matrix(self.m, self.x, mu, norm="one")  # (4, 4, n_mu)
        s11 = p[0, 0]

        # CDF of cos(theta): no extra sin(theta) factor needed since phi is
        # sampled uniformly elsewhere, so dOmega = dphi d(cos theta) makes
        # S11 itself the marginal density in mu.
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (s11[1:] + s11[:-1]) * np.diff(mu))])
        cdf /= cdf[-1]

        self._mu = torch.as_tensor(mu, dtype=torch.float64)
        self._cdf = torch.as_tensor(cdf, dtype=torch.float64)
        self._mueller_table = torch.as_tensor(np.moveaxis(p, -1, 0).copy(), dtype=torch.float64)  # (n_mu, 4, 4)

    def to(self, device: torch.device, dtype: torch.dtype) -> "Mie":
        """Move the precomputed tables to the simulation's device/dtype."""
        self._mu = self._mu.to(device=device, dtype=dtype)
        self._cdf = self._cdf.to(device=device, dtype=dtype)
        self._mueller_table = self._mueller_table.to(device=device, dtype=dtype)
        return self

    def sample_cos_theta(self, xi: torch.Tensor) -> torch.Tensor:
        """Invert the tabulated Mie CDF to draw polar scattering cosines."""
        return _interp_1d(self._cdf, self._mu, xi)

    def mueller_matrix(self, cos_theta: torch.Tensor) -> torch.Tensor:
        """Batched Mie Mueller matrices, shape (N, 4, 4), at the given cos(theta)."""
        idx = torch.bucketize(cos_theta, self._mu).clamp(1, len(self._mu) - 1)
        lo, hi = idx - 1, idx
        mu_lo, mu_hi = self._mu[lo], self._mu[hi]
        frac = ((cos_theta - mu_lo) / (mu_hi - mu_lo).clamp_min(1e-12)).clamp(0.0, 1.0)

        table = self._mueller_table.reshape(len(self._mu), 16)
        m = table[lo] + frac.unsqueeze(-1) * (table[hi] - table[lo])
        return m.reshape(-1, 4, 4)


def _interp_1d(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Batched 1D linear interpolation; `xp` must be sorted ascending."""
    idx = torch.bucketize(x, xp).clamp(1, len(xp) - 1)
    lo, hi = idx - 1, idx
    x0, x1 = xp[lo], xp[hi]
    frac = ((x - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
    return fp[lo] + frac * (fp[hi] - fp[lo])
