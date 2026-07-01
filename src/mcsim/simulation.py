"""Monte Carlo photon transport simulation (MCML-style).

Reference: Wang, Jacques & Zheng, Comp Meth Prog Biomed 47 (1995) 131-146.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import miepython
import numpy as np
from numpy.random import Generator
from tqdm import tqdm

from .tissue import TissueLayer

_WEIGHT_THRESHOLD = 1e-4
_ROULETTE_SURVIVE = 0.1
_MIE_ANGLE_BINS = 721  # 0.25 deg resolution over cos(theta) in [-1, 1]
_N_POLARIZATION_BINS = 45  # exit-angle bins spanning 0-90 deg


@dataclass
class SimulationResult:
    n_photons: int
    reflectance: float
    transmittance: float
    absorbed: float
    specular_r: float
    z_bins: np.ndarray
    absorbed_profile: np.ndarray
    exit_angle_bins: np.ndarray
    dolp_profile: np.ndarray

    def check_energy_conservation(self, tol: float = 1e-3) -> bool:
        total = self.reflectance + self.transmittance + self.absorbed + self.specular_r
        return abs(total - 1.0) < tol


class Simulation:
    """Run a Monte Carlo simulation for a single homogeneous tissue slab.

    Parameters
    ----------
    layer           : TissueLayer describing the medium.
    n_above         : refractive index of the medium above the slab (default air=1.0).
    n_below         : refractive index of the medium below the slab (default air=1.0).
    n_bins          : number of depth bins for the absorbed-energy profile.
    seed            : RNG seed for reproducibility.
    initial_stokes  : Stokes vector [I, Q, U, V] of the incident photons, defined
                      in the (x, y) lab frame perpendicular to the beam. Defaults
                      to unpolarised light.
    """

    def __init__(
        self,
        layer: TissueLayer,
        n_above: float = 1.0,
        n_below: float = 1.0,
        n_bins: int = 200,
        seed: int | None = 42,
        initial_stokes: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.layer = layer
        self.n_above = n_above
        self.n_below = n_below
        self.n_bins = n_bins
        self.rng: Generator = np.random.default_rng(seed)
        self.initial_stokes = tuple(float(v) for v in initial_stokes)

        # Depth binning over the slab (or 10 mm for semi-infinite)
        depth = layer.depth if math.isfinite(layer.depth) else 10.0
        self.z_bins = np.linspace(0.0, depth, n_bins + 1)

        # Exit-angle binning (degrees from the surface normal) for the
        # degree-of-linear-polarization profile.
        self.n_pol_bins = _N_POLARIZATION_BINS
        self.exit_angle_bins = np.linspace(0.0, 90.0, self.n_pol_bins + 1)

        # Precompute the Mie scattering (Mueller) matrix on a fine grid of
        # scattering angles, so each scattering event can look up the matrix
        # for the same deflection angle it samples from the Henyey-Greenstein
        # phase function, without re-running the Mie series every event.
        # Stored as plain Python tuples (rather than indexing a numpy array
        # per event) since this lookup sits in the hot per-scatter-event loop.
        mu_grid = np.linspace(-1.0, 1.0, _MIE_ANGLE_BINS)
        grid = miepython.phase_matrix(
            complex(layer.mie_relative_index, 0.0),
            layer.mie_size_parameter,
            mu_grid,
        )  # shape (4, 4, _MIE_ANGLE_BINS)
        self._mueller_grid: list[tuple[float, ...]] = [
            tuple(grid[:, :, i].reshape(-1)) for i in range(_MIE_ANGLE_BINS)
        ]

    def _mueller_matrix(self, cos_theta: float) -> tuple[float, ...]:
        """Look up the precomputed (flattened, row-major) Mueller matrix for a scattering angle."""
        idx = int(round((cos_theta + 1.0) * 0.5 * (_MIE_ANGLE_BINS - 1)))
        idx = min(max(idx, 0), _MIE_ANGLE_BINS - 1)
        return self._mueller_grid[idx]

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

        # Weighted Stokes-vector sums per exit-angle bin, used to compute the
        # ensemble degree of linear polarization as a function of exit angle.
        pol_i = np.zeros(self.n_pol_bins)
        pol_q = np.zeros(self.n_pol_bins)
        pol_u = np.zeros(self.n_pol_bins)

        for _ in tqdm(range(n_photons), desc="Photons", unit="photon"):
            dr, dt, da, prof, exit_angle, exit_stokes, exit_weight = self._trace_photon(r_specular)
            total_r += dr
            total_t += dt
            total_a += da
            abs_profile += prof
            if exit_angle is not None:
                bin_idx = min(int(exit_angle / 90.0 * self.n_pol_bins), self.n_pol_bins - 1)
                pol_i[bin_idx] += exit_weight * exit_stokes[0]
                pol_q[bin_idx] += exit_weight * exit_stokes[1]
                pol_u[bin_idx] += exit_weight * exit_stokes[2]

        dolp_profile = np.zeros(self.n_pol_bins)
        has_signal = pol_i > 0
        dolp_profile[has_signal] = (
            np.sqrt(pol_q[has_signal] ** 2 + pol_u[has_signal] ** 2) / pol_i[has_signal]
        )

        scale = 1.0 / n_photons
        return SimulationResult(
            n_photons=n_photons,
            reflectance=total_r * scale,
            transmittance=total_t * scale,
            absorbed=total_a * scale,
            specular_r=r_specular,
            z_bins=self.z_bins,
            absorbed_profile=abs_profile * scale,
            exit_angle_bins=self.exit_angle_bins,
            dolp_profile=dolp_profile,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trace_photon(
        self, r_specular: float
    ) -> tuple[float, float, float, np.ndarray, float | None, tuple[float, ...] | None, float]:
        layer = self.layer
        rng = self.rng

        # Photon state: position, propagation direction (ux, uy, uz), and the
        # local polarization reference frame (frame_par, frame_perp), which
        # together with the direction forms a right-handed orthonormal
        # triad. The Stokes vector is always expressed in this local frame.
        # Plain tuples/floats (rather than numpy arrays) are used here since
        # this is the hot per-scatter-event loop.
        x, y, z = 0.0, 0.0, 0.0
        u = (0.0, 0.0, 1.0)
        frame_par = (1.0, 0.0, 0.0)
        frame_perp = (0.0, 1.0, 0.0)
        stokes = self.initial_stokes
        weight = 1.0 - r_specular            # subtract specular at entry

        total_r = 0.0
        total_t = 0.0
        abs_profile = np.zeros(self.n_bins)
        exit_angle: float | None = None
        exit_stokes: tuple[float, ...] | None = None
        exit_weight = 0.0

        while weight > 0.0:
            # --- Step ---
            xi = rng.random()
            # avoid log(0) — xi=0 is vanishingly rare but possible
            if xi == 0.0:
                xi = 1e-300
            step = -math.log(xi) / layer.mu_t

            uz = u[2]

            # Move photon
            x_new = x + u[0] * step
            y_new = y + u[1] * step
            z_new = z + uz * step

            # --- Boundary check ---
            if z_new < 0.0:
                # Photon hit top surface; check internal reflection
                t_frac = -z / (uz * step)         # fraction of step to boundary
                z_cross = 0.0
                if _internal_reflection(layer.n, self.n_above, uz, rng):
                    # Reflect: mirror the z-component of direction and frame,
                    # keeping (u, frame_par, frame_perp) orthonormal.
                    z = z_cross
                    x += u[0] * step * t_frac
                    y += u[1] * step * t_frac
                    u = (u[0], u[1], -u[2])
                    frame_par = (frame_par[0], frame_par[1], -frame_par[2])
                    frame_perp = (frame_perp[0], frame_perp[1], -frame_perp[2])
                    continue
                else:
                    total_r += weight
                    exit_angle = math.degrees(math.acos(min(1.0, abs(uz))))
                    exit_stokes = stokes
                    exit_weight = weight
                    break

            elif z_new > layer.depth:
                # Photon hit bottom surface
                if _internal_reflection(layer.n, self.n_below, uz, rng):
                    t_frac = (layer.depth - z) / (uz * step)
                    z = layer.depth
                    x += u[0] * step * t_frac
                    y += u[1] * step * t_frac
                    u = (u[0], u[1], -u[2])
                    frame_par = (frame_par[0], frame_par[1], -frame_par[2])
                    frame_perp = (frame_perp[0], frame_perp[1], -frame_perp[2])
                    continue
                else:
                    total_t += weight
                    exit_angle = math.degrees(math.acos(min(1.0, abs(uz))))
                    exit_stokes = stokes
                    exit_weight = weight
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
            # Sample the same deflection angle used for both the photon's
            # new direction and the Mueller-matrix lookup, and a fresh
            # azimuth about the current direction.
            cos_theta = _sample_hg_cos_theta(layer.g, rng)
            phi = 2.0 * math.pi * rng.random()

            u, frame_par, frame_perp = _rotate_frame(u, frame_par, frame_perp, cos_theta, phi)
            stokes = _rotate_stokes(stokes, phi)
            stokes = _apply_mueller(self._mueller_matrix(cos_theta), stokes)
            if stokes[0] > 0.0:
                inv_i = 1.0 / stokes[0]
                stokes = (stokes[0] * inv_i, stokes[1] * inv_i, stokes[2] * inv_i, stokes[3] * inv_i)

        total_a = abs_profile.sum()
        return total_r, total_t, total_a, abs_profile, exit_angle, exit_stokes, exit_weight


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


def _sample_hg_cos_theta(g: float, rng: Generator) -> float:
    """Sample the scattering deflection angle from the Henyey-Greenstein phase function."""
    xi1 = rng.random()
    if abs(g) < 1e-10:
        cos_theta = 2.0 * xi1 - 1.0
    else:
        tmp = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi1)
        cos_theta = (1.0 + g * g - tmp * tmp) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))
    return cos_theta


Vec3 = tuple[float, float, float]


def _rotate_frame(
    u: Vec3,
    frame_par: Vec3,
    frame_perp: Vec3,
    cos_theta: float,
    phi: float,
) -> tuple[Vec3, Vec3, Vec3]:
    """Deflect the propagation direction and update the local reference frame.

    ``frame_par``/``frame_perp`` are unit vectors perpendicular to ``u`` such
    that (u, frame_par, frame_perp) is a right-handed orthonormal triad. The
    photon is first rotated in azimuth by ``phi`` about ``u`` (bringing
    ``frame_par`` into the new scattering plane), then deflected by the polar
    angle ``theta`` within that plane. This mirrors the rotation applied to
    the photon's Stokes vector by ``_rotate_stokes`` and the Mueller matrix,
    so the direction update and the polarization update always use the same
    (theta, phi) pair.
    """
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    par_rx = cos_phi * frame_par[0] + sin_phi * frame_perp[0]
    par_ry = cos_phi * frame_par[1] + sin_phi * frame_perp[1]
    par_rz = cos_phi * frame_par[2] + sin_phi * frame_perp[2]

    perp_new = (
        -sin_phi * frame_par[0] + cos_phi * frame_perp[0],
        -sin_phi * frame_par[1] + cos_phi * frame_perp[1],
        -sin_phi * frame_par[2] + cos_phi * frame_perp[2],
    )

    ux = cos_theta * u[0] + sin_theta * par_rx
    uy = cos_theta * u[1] + sin_theta * par_ry
    uz = cos_theta * u[2] + sin_theta * par_rz
    u_norm = math.sqrt(ux * ux + uy * uy + uz * uz)
    u_new = (ux / u_norm, uy / u_norm, uz / u_norm)

    pux = cos_theta * par_rx - sin_theta * u[0]
    puy = cos_theta * par_ry - sin_theta * u[1]
    puz = cos_theta * par_rz - sin_theta * u[2]
    p_norm = math.sqrt(pux * pux + puy * puy + puz * puz)
    par_new = (pux / p_norm, puy / p_norm, puz / p_norm)

    return u_new, par_new, perp_new


def _rotate_stokes(stokes: tuple[float, float, float, float], phi: float) -> tuple[float, float, float, float]:
    """Rotate a Stokes vector's reference frame by azimuth ``phi`` about the propagation axis."""
    cos_2phi = math.cos(2.0 * phi)
    sin_2phi = math.sin(2.0 * phi)
    i, q, u, v = stokes
    return (i, cos_2phi * q + sin_2phi * u, -sin_2phi * q + cos_2phi * u, v)


def _apply_mueller(
    matrix: tuple[float, ...], stokes: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Apply a 4x4 Mueller matrix (flattened, row-major) to a Stokes vector."""
    m = matrix
    i, q, u, v = stokes
    return (
        m[0] * i + m[1] * q + m[2] * u + m[3] * v,
        m[4] * i + m[5] * q + m[6] * u + m[7] * v,
        m[8] * i + m[9] * q + m[10] * u + m[11] * v,
        m[12] * i + m[13] * q + m[14] * u + m[15] * v,
    )
