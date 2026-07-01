"""Monte Carlo photon transport simulation (MCML-style), vectorized with torch.

All photons in a batch are traced simultaneously as torch tensors (one row per
photon) so the simulation can run on a GPU; each iteration of the loop below
advances every still-alive photon by one scattering event (or one boundary
interaction) at once, using boolean masks in place of the branches an
unbatched, per-photon implementation would use.

Reference: Wang, Jacques & Zheng, Comp Meth Prog Biomed 47 (1995) 131-146.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import miepython
import numpy as np
import torch
from tqdm import tqdm

from .tissue import TissueLayer

_WEIGHT_THRESHOLD = 1e-4
_ROULETTE_SURVIVE = 0.1
_MIE_ANGLE_BINS = 721  # 0.25 deg resolution over cos(theta) in [-1, 1]
_N_POLARIZATION_BINS = 45  # exit-angle bins spanning 0-90 deg
_DTYPE = torch.float64


def _select_device(preferred: str | None) -> torch.device:
    """Pick a torch device: the requested one, else GPU if available, else CPU."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


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
    """Run a batched Monte Carlo simulation for a single homogeneous tissue slab.

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
    device          : torch device to run on ("cuda", "cpu", ...). Defaults to
                      GPU when available, otherwise CPU. If running on the
                      selected device raises an error, ``run`` automatically
                      retries on CPU.
    """

    def __init__(
        self,
        layer: TissueLayer,
        n_above: float = 1.0,
        n_below: float = 1.0,
        n_bins: int = 200,
        seed: int | None = 42,
        initial_stokes: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        device: str | None = None,
    ) -> None:
        self.layer = layer
        self.n_above = n_above
        self.n_below = n_below
        self.n_bins = n_bins
        self.seed = seed
        self.initial_stokes = tuple(float(v) for v in initial_stokes)
        self.device = _select_device(device)

        # Depth binning over the slab (or 10 mm for semi-infinite)
        self._bin_depth = layer.depth if math.isfinite(layer.depth) else 10.0
        self.z_bins = np.linspace(0.0, self._bin_depth, n_bins + 1)

        # Exit-angle binning (degrees from the surface normal) for the
        # degree-of-linear-polarization profile.
        self.n_pol_bins = _N_POLARIZATION_BINS
        self.exit_angle_bins = np.linspace(0.0, 90.0, self.n_pol_bins + 1)

        # Precompute the Mie scattering (Mueller) matrix on a fine grid of
        # scattering angles, so each scattering event can look up the matrix
        # for the same deflection angle it samples from the Henyey-Greenstein
        # phase function, without re-running the Mie series every event.
        mu_grid = np.linspace(-1.0, 1.0, _MIE_ANGLE_BINS)
        grid = miepython.phase_matrix(
            complex(layer.mie_relative_index, 0.0),
            layer.mie_size_parameter,
            mu_grid,
        )  # shape (4, 4, _MIE_ANGLE_BINS)
        self._mueller_grid_np = np.ascontiguousarray(np.moveaxis(grid, 2, 0))  # (_MIE_ANGLE_BINS, 4, 4)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, n_photons: int) -> SimulationResult:
        """Run the simulation, preferring ``self.device`` and falling back to CPU on error."""
        try:
            return self._run_on_device(n_photons, self.device)
        except Exception as exc:
            if self.device.type == "cpu":
                raise
            warnings.warn(f"Simulation on {self.device} failed ({exc!r}); falling back to CPU.")
            return self._run_on_device(n_photons, torch.device("cpu"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_on_device(self, n_photons: int, device: torch.device) -> SimulationResult:
        layer = self.layer
        n = n_photons

        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed if self.seed is not None else torch.seed())

        def rand(*shape: int) -> torch.Tensor:
            return torch.rand(*shape, generator=gen, device=device, dtype=_DTYPE)

        mueller_grid = torch.as_tensor(self._mueller_grid_np, dtype=_DTYPE, device=device)

        r_specular = _fresnel_specular(self.n_above, layer.n)
        mu_t = layer.mu_t
        mu_a = layer.mu_a
        g = layer.g
        layer_depth = layer.depth  # may be math.inf for a semi-infinite slab
        bin_depth = self._bin_depth
        n_bins = self.n_bins
        n_pol_bins = self.n_pol_bins

        # Photon state, one row per photon. (u, frame_par, frame_perp) form a
        # right-handed orthonormal triad; the Stokes vector is expressed in it.
        pos = torch.zeros((n, 3), dtype=_DTYPE, device=device)
        direction = torch.zeros((n, 3), dtype=_DTYPE, device=device)
        direction[:, 2] = 1.0
        frame_par = torch.zeros((n, 3), dtype=_DTYPE, device=device)
        frame_par[:, 0] = 1.0
        frame_perp = torch.zeros((n, 3), dtype=_DTYPE, device=device)
        frame_perp[:, 1] = 1.0
        stokes = torch.tensor(self.initial_stokes, dtype=_DTYPE, device=device).expand(n, 4).clone()
        weight = torch.full((n,), 1.0 - r_specular, dtype=_DTYPE, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)

        total_r = torch.zeros((), dtype=_DTYPE, device=device)
        total_t = torch.zeros((), dtype=_DTYPE, device=device)
        abs_profile = torch.zeros(n_bins, dtype=_DTYPE, device=device)
        pol_i = torch.zeros(n_pol_bins, dtype=_DTYPE, device=device)
        pol_q = torch.zeros(n_pol_bins, dtype=_DTYPE, device=device)
        pol_u = torch.zeros(n_pol_bins, dtype=_DTYPE, device=device)

        progress = tqdm(total=n, desc="Photons", unit="photon")
        n_alive_prev = n

        while bool(alive.any()):
            # --- Step ---
            xi = rand(n).clamp_min(1e-300)
            step = -torch.log(xi) / mu_t

            uz = direction[:, 2]
            x_new = pos[:, 0] + direction[:, 0] * step
            y_new = pos[:, 1] + direction[:, 1] * step
            z_new = pos[:, 2] + uz * step

            # --- Boundary classification ---
            crossing_top = alive & (z_new < 0.0)
            crossing_bottom = alive & (z_new > layer_depth)
            no_cross = alive & ~crossing_top & ~crossing_bottom

            cos_i = uz.abs()
            r_top = _fresnel_reflectance(layer.n, self.n_above, cos_i)
            r_bottom = _fresnel_reflectance(layer.n, self.n_below, cos_i)
            reflect_top = crossing_top & (rand(n) < r_top)
            reflect_bottom = crossing_bottom & (rand(n) < r_bottom)
            reflect_mask = reflect_top | reflect_bottom
            exit_top = crossing_top & ~reflect_top
            exit_bottom = crossing_bottom & ~reflect_bottom
            exit_mask = exit_top | exit_bottom

            # --- Reflection geometry: land exactly on the crossed boundary ---
            t_frac_top = -pos[:, 2] / (uz * step)
            t_frac_bottom = (layer_depth - pos[:, 2]) / (uz * step)
            t_frac = torch.where(crossing_top, t_frac_top, t_frac_bottom)
            z_boundary = torch.where(crossing_top, torch.zeros_like(pos[:, 2]), torch.full_like(pos[:, 2], bin_depth))
            refl_x = pos[:, 0] + direction[:, 0] * step * t_frac
            refl_y = pos[:, 1] + direction[:, 1] * step * t_frac

            # --- Exiting photons: accumulate R/T and polarization-vs-angle bins ---
            exit_angle_deg = torch.rad2deg(torch.acos(cos_i.clamp(max=1.0)))
            bin_idx_pol = (exit_angle_deg / 90.0 * n_pol_bins).long().clamp(0, n_pol_bins - 1)

            total_r = total_r + weight[exit_top].sum()
            total_t = total_t + weight[exit_bottom].sum()
            if exit_mask.any():
                idx_sel = bin_idx_pol[exit_mask]
                w_sel = weight[exit_mask]
                s_sel = stokes[exit_mask]
                pol_i.index_add_(0, idx_sel, w_sel * s_sel[:, 0])
                pol_q.index_add_(0, idx_sel, w_sel * s_sel[:, 1])
                pol_u.index_add_(0, idx_sel, w_sel * s_sel[:, 2])

            # --- Absorption (only for photons that stayed inside this step) ---
            delta_w = weight * (mu_a / mu_t)
            weight_after_abs = torch.where(no_cross, weight - delta_w, weight)
            if no_cross.any():
                bin_idx_abs = (z_new / bin_depth * n_bins).long().clamp(0, n_bins - 1)
                abs_profile.index_add_(0, bin_idx_abs[no_cross], delta_w[no_cross])

            # --- Russian roulette ---
            below_threshold = no_cross & (weight_after_abs < _WEIGHT_THRESHOLD)
            survive = below_threshold & (rand(n) < _ROULETTE_SURVIVE)
            died = below_threshold & ~survive
            weight_after_roulette = torch.where(survive, weight_after_abs / _ROULETTE_SURVIVE, weight_after_abs)

            # --- Scatter: sample the same (theta, phi) for direction and Mueller lookup ---
            scatter_mask = no_cross & ~died
            xi_hg = rand(n)
            if abs(g) < 1e-10:
                cos_theta = 2.0 * xi_hg - 1.0
            else:
                tmp = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi_hg)
                cos_theta = (1.0 + g * g - tmp * tmp) / (2.0 * g)
                cos_theta = cos_theta.clamp(-1.0, 1.0)
            phi = 2.0 * math.pi * rand(n)

            new_dir, new_par, new_perp = _rotate_frame_batch(direction, frame_par, frame_perp, cos_theta, phi)
            rotated_stokes = _rotate_stokes_batch(stokes, phi)
            idx_mie = ((cos_theta + 1.0) * 0.5 * (_MIE_ANGLE_BINS - 1)).round().long().clamp(0, _MIE_ANGLE_BINS - 1)
            matrices = mueller_grid[idx_mie]  # (n, 4, 4)
            scattered_stokes = torch.bmm(matrices, rotated_stokes.unsqueeze(-1)).squeeze(-1)
            new_i = scattered_stokes[:, 0]
            safe = (new_i > 0.0).unsqueeze(-1)
            scattered_stokes = torch.where(safe, scattered_stokes / new_i.clamp_min(1e-300).unsqueeze(-1), scattered_stokes)

            # --- Combine branches into the next state ---
            mirror = torch.tensor([1.0, 1.0, -1.0], dtype=_DTYPE, device=device)
            pos = torch.stack(
                [
                    torch.where(reflect_mask, refl_x, torch.where(no_cross, x_new, pos[:, 0])),
                    torch.where(reflect_mask, refl_y, torch.where(no_cross, y_new, pos[:, 1])),
                    torch.where(reflect_mask, z_boundary, torch.where(no_cross, z_new, pos[:, 2])),
                ],
                dim=1,
            )
            reflect_col = reflect_mask.unsqueeze(-1)
            scatter_col = scatter_mask.unsqueeze(-1)
            direction = torch.where(reflect_col, direction * mirror, direction)
            direction = torch.where(scatter_col, new_dir, direction)
            frame_par = torch.where(reflect_col, frame_par * mirror, frame_par)
            frame_par = torch.where(scatter_col, new_par, frame_par)
            frame_perp = torch.where(reflect_col, frame_perp * mirror, frame_perp)
            frame_perp = torch.where(scatter_col, new_perp, frame_perp)
            stokes = torch.where(scatter_col, scattered_stokes, stokes)
            weight = torch.where(no_cross, weight_after_roulette, weight)

            alive = alive & ~exit_mask & ~died

            n_alive = int(alive.sum().item())
            progress.update(n_alive_prev - n_alive)
            n_alive_prev = n_alive

        progress.close()

        dolp_profile = torch.zeros(n_pol_bins, dtype=_DTYPE, device=device)
        has_signal = pol_i > 0
        dolp_profile[has_signal] = torch.sqrt(pol_q[has_signal] ** 2 + pol_u[has_signal] ** 2) / pol_i[has_signal]

        scale = 1.0 / n_photons
        return SimulationResult(
            n_photons=n_photons,
            reflectance=total_r.item() * scale,
            transmittance=total_t.item() * scale,
            absorbed=abs_profile.sum().item() * scale,
            specular_r=r_specular,
            z_bins=self.z_bins,
            absorbed_profile=abs_profile.cpu().numpy() * scale,
            exit_angle_bins=self.exit_angle_bins,
            dolp_profile=dolp_profile.cpu().numpy(),
        )


# ------------------------------------------------------------------
# Physics functions
# ------------------------------------------------------------------

def _fresnel_specular(n_i: float, n_t: float) -> float:
    """Normal-incidence specular reflectance at a planar interface."""
    return ((n_i - n_t) / (n_i + n_t)) ** 2


def _fresnel_reflectance(n_i: float, n_t: float, cos_i: torch.Tensor) -> torch.Tensor:
    """Batched Fresnel reflectance probability (1.0 under total internal reflection)."""
    cos_i = cos_i.clamp(0.0, 1.0)
    sin_i2 = (1.0 - cos_i**2).clamp_min(0.0)

    if n_t < n_i:
        cos_c2 = 1.0 - (n_t / n_i) ** 2
        tir = cos_i**2 < cos_c2
    else:
        tir = torch.zeros_like(cos_i, dtype=torch.bool)

    sin_t = n_i * torch.sqrt(sin_i2) / n_t
    beyond_critical = sin_t >= 1.0
    cos_t = torch.sqrt((1.0 - sin_t.clamp(max=1.0 - 1e-12) ** 2).clamp_min(0.0))

    rs = ((n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)) ** 2
    rp = ((n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t)) ** 2
    r_fresnel = 0.5 * (rs + rp)

    return torch.where(tir | beyond_critical, torch.ones_like(cos_i), r_fresnel)


def _rotate_frame_batch(
    u: torch.Tensor,
    frame_par: torch.Tensor,
    frame_perp: torch.Tensor,
    cos_theta: torch.Tensor,
    phi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched version of the single-photon frame/direction rotation.

    ``frame_par``/``frame_perp`` are unit vectors perpendicular to ``u`` such
    that (u, frame_par, frame_perp) is a right-handed orthonormal triad. The
    photon is first rotated in azimuth by ``phi`` about ``u`` (bringing
    ``frame_par`` into the new scattering plane), then deflected by the polar
    angle ``theta`` within that plane. This mirrors the rotation applied to
    the photon's Stokes vector by ``_rotate_stokes_batch`` and the Mueller
    matrix, so the direction update and the polarization update always use
    the same (theta, phi) pair.
    """
    sin_theta = torch.sqrt((1.0 - cos_theta**2).clamp_min(0.0)).unsqueeze(-1)
    cos_theta = cos_theta.unsqueeze(-1)
    cos_phi = torch.cos(phi).unsqueeze(-1)
    sin_phi = torch.sin(phi).unsqueeze(-1)

    par_r = cos_phi * frame_par + sin_phi * frame_perp
    perp_new = -sin_phi * frame_par + cos_phi * frame_perp

    u_new = cos_theta * u + sin_theta * par_r
    u_new = u_new / u_new.norm(dim=-1, keepdim=True)

    par_new = cos_theta * par_r - sin_theta * u
    par_new = par_new / par_new.norm(dim=-1, keepdim=True)

    perp_new = perp_new / perp_new.norm(dim=-1, keepdim=True)

    return u_new, par_new, perp_new


def _rotate_stokes_batch(stokes: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Batched rotation of a Stokes vector's reference frame by azimuth ``phi``."""
    cos_2phi = torch.cos(2.0 * phi)
    sin_2phi = torch.sin(2.0 * phi)
    i, q, u, v = stokes.unbind(-1)
    return torch.stack([i, cos_2phi * q + sin_2phi * u, -sin_2phi * q + cos_2phi * u, v], dim=-1)
