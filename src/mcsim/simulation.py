"""Monte Carlo photon transport simulation (MCML-style).

Reference: Wang, Jacques & Zheng, Comp Meth Prog Biomed 47 (1995) 131-146.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator
from tqdm import tqdm

from .tissue import TissueLayer

_WEIGHT_THRESHOLD = 1e-4
_ROULETTE_SURVIVE = 0.1
_ON_AXIS_TOL = 1.0 - 1e-12


@dataclass
class SimulationResult:
    n_photons: int
    reflectance: float
    transmittance: float
    absorbed: float
    specular_r: float
    z_bins: np.ndarray
    absorbed_profile: np.ndarray

    def check_energy_conservation(self, tol: float = 1e-3) -> bool:
        total = self.reflectance + self.transmittance + self.absorbed + self.specular_r
        return abs(total - 1.0) < tol


class Simulation:
    """Run a Monte Carlo simulation for a single homogeneous tissue slab.

    Parameters
    ----------
    layer   : TissueLayer describing the medium.
    n_above : refractive index of the medium above the slab (default air=1.0).
    n_below : refractive index of the medium below the slab (default air=1.0).
    n_bins  : number of depth bins for the absorbed-energy profile.
    seed    : RNG seed for reproducibility.
    """

    def __init__(
        self,
        layer: TissueLayer,
        n_above: float = 1.0,
        n_below: float = 1.0,
        n_bins: int = 200,
        seed: int | None = 42,
    ) -> None:
        self.layer = layer
        self.n_above = n_above
        self.n_below = n_below
        self.n_bins = n_bins
        self.rng: Generator = np.random.default_rng(seed)

        # Depth binning over the slab (or 10 mm for semi-infinite)
        depth = layer.depth if math.isfinite(layer.depth) else 10.0
        self.z_bins = np.linspace(0.0, depth, n_bins + 1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, n_photons: int) -> SimulationResult:
        layer = self.layer

        # Specular reflectance at the top surface (normal incidence)
        r_specular = _fresnel_specular(self.n_above, layer.n)

        total_r = 0.0      # diffuse reflectance accumulator
        total_t = 0.0      # transmittance accumulator
        total_a = 0.0      # absorbed weight accumulator
        abs_profile = np.zeros(self.n_bins)

        for _ in tqdm(range(n_photons), desc="Photons", unit="photon"):
            dr, dt, da, prof = self._trace_photon(r_specular)
            total_r += dr
            total_t += dt
            total_a += da
            abs_profile += prof

        scale = 1.0 / n_photons
        return SimulationResult(
            n_photons=n_photons,
            reflectance=total_r * scale,
            transmittance=total_t * scale,
            absorbed=total_a * scale,
            specular_r=r_specular,
            z_bins=self.z_bins,
            absorbed_profile=abs_profile * scale,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trace_photon(
        self, r_specular: float
    ) -> tuple[float, float, float, np.ndarray]:
        layer = self.layer
        rng = self.rng

        # Photon state
        x, y, z = 0.0, 0.0, 0.0
        ux, uy, uz = 0.0, 0.0, 1.0          # direction cosines
        weight = 1.0 - r_specular            # subtract specular at entry

        total_r = 0.0
        total_t = 0.0
        abs_profile = np.zeros(self.n_bins)

        while weight > 0.0:
            # --- Step ---
            xi = rng.random()
            # avoid log(0) — xi=0 is vanishingly rare but possible
            if xi == 0.0:
                xi = 1e-300
            step = -math.log(xi) / layer.mu_t

            # Move photon
            x_new = x + ux * step
            y_new = y + uy * step
            z_new = z + uz * step

            # --- Boundary check ---
            if z_new < 0.0:
                # Photon hit top surface; check internal reflection
                t_frac = -z / (uz * step)         # fraction of step to boundary
                z_cross = 0.0
                if _internal_reflection(layer.n, self.n_above, uz, rng):
                    # Reflect: reverse z-component, stay inside
                    z = z_cross
                    x += ux * step * t_frac
                    y += uy * step * t_frac
                    uz = -uz
                    continue
                else:
                    total_r += weight
                    break

            elif z_new > layer.depth:
                # Photon hit bottom surface
                if _internal_reflection(layer.n, self.n_below, uz, rng):
                    t_frac = (layer.depth - z) / (uz * step)
                    z = layer.depth
                    x += ux * step * t_frac
                    y += uy * step * t_frac
                    uz = -uz
                    continue
                else:
                    total_t += weight
                    break

            x, y, z = x_new, y_new, z_new

            # --- Absorption ---
            delta_w = weight * (layer.mu_a / layer.mu_t)
            weight -= delta_w

            # Deposit into depth profile
            bin_idx = int(z / layer.depth * self.n_bins) if math.isfinite(layer.depth) else int(z / 10.0 * self.n_bins)
            bin_idx = min(bin_idx, self.n_bins - 1)
            abs_profile[bin_idx] += delta_w

            # --- Russian roulette ---
            if weight < _WEIGHT_THRESHOLD:
                if rng.random() < _ROULETTE_SURVIVE:
                    weight /= _ROULETTE_SURVIVE
                else:
                    break

            # --- Scatter ---
            ux, uy, uz = _henyey_greenstein_scatter(ux, uy, uz, layer.g, rng)

        total_a = abs_profile.sum()
        return total_r, total_t, total_a, abs_profile


# ------------------------------------------------------------------
# Physics functions
# ------------------------------------------------------------------

def _fresnel_specular(n_i: float, n_t: float) -> float:
    """Normal-incidence specular reflectance at a planar interface."""
    return ((n_i - n_t) / (n_i + n_t)) ** 2


def _critical_angle_cos(n_i: float, n_t: float) -> float:
    """cos(theta_c) for total internal reflection; 0 if n_t >= n_i."""
    if n_t >= n_i:
        return 0.0
    return math.sqrt(1.0 - (n_t / n_i) ** 2)


def _internal_reflection(n_i: float, n_t: float, uz: float, rng: Generator) -> bool:
    """Return True if the photon is internally reflected at a boundary.

    Uses Fresnel reflectance for unpolarised light.  Photon travels in the
    +z direction so uz > 0 for a bottom surface hit, uz < 0 for a top hit.
    """
    cos_i = abs(uz)

    # Total internal reflection
    if n_t < n_i:
        cos_c = _critical_angle_cos(n_i, n_t)
        if cos_i < cos_c:
            return True

    # Fresnel probabilistic reflection
    sin_i = math.sqrt(max(0.0, 1.0 - cos_i**2))
    sin_t = n_i * sin_i / n_t
    if sin_t >= 1.0:
        return True
    cos_t = math.sqrt(max(0.0, 1.0 - sin_t**2))

    rs = ((n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)) ** 2
    rp = ((n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t)) ** 2
    r_fres = 0.5 * (rs + rp)

    return rng.random() < r_fres


def _henyey_greenstein_scatter(
    ux: float, uy: float, uz: float, g: float, rng: Generator
) -> tuple[float, float, float]:
    """Update direction cosines using Henyey-Greenstein phase function."""
    xi1 = rng.random()
    xi2 = rng.random()

    # Deflection angle cos(theta)
    if abs(g) < 1e-10:
        cos_theta = 2.0 * xi1 - 1.0
    else:
        tmp = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi1)
        cos_theta = (1.0 + g * g - tmp * tmp) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))

    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
    phi = 2.0 * math.pi * xi2
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    # Direction cosine update (special case when beam is nearly on-axis)
    if abs(uz) > _ON_AXIS_TOL:
        sign_z = 1.0 if uz > 0 else -1.0
        ux_new = sin_theta * cos_phi
        uy_new = sin_theta * sin_phi
        uz_new = sign_z * cos_theta
    else:
        denom = math.sqrt(1.0 - uz**2)
        ux_new = sin_theta * (ux * uz * cos_phi - uy * sin_phi) / denom + ux * cos_theta
        uy_new = sin_theta * (uy * uz * cos_phi + ux * sin_phi) / denom + uy * cos_theta
        uz_new = -sin_theta * cos_phi * denom + uz * cos_theta

    # Normalise to guard against floating-point drift
    norm = math.sqrt(ux_new**2 + uy_new**2 + uz_new**2)
    return ux_new / norm, uy_new / norm, uz_new / norm
