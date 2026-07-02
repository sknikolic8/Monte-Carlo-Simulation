"""Vectorized Monte Carlo photon transport simulator.

See CLAUDE.md for the full physical model (step sampling, boundary Fresnel
interactions, polarization tracking). All photons are advanced together as
batched tensors instead of looping photon-by-photon in Python, so the whole
population is propagated with a fixed number of large tensor ops per step
regardless of N.
"""

from __future__ import annotations

import math

import torch
from tqdm import tqdm

from .medium import Medium
from .params import (
    DEVICE, DTYPE, INITIAL_STOKES, MAX_STEPS, RNG_SEED, SURVIVAL_CHANCE, USE_POLARIZED_SAMPLING, WEIGHT_THRESHOLD,
)
from .phase_functions.mueller import fresnel_mueller_matrices, rayleigh_mueller_matrix, rotate_stokes


class MonteCarlo:
    """Vectorized Monte Carlo photon transport simulator.

    All photons are advanced together as batched tensors instead of looping
    photon-by-photon in Python, so the entire population is propagated with
    a fixed number of large tensor ops per step regardless of N.
    """

    def __init__(self, medium: Medium, phase_function, n_photons: int, device: str = DEVICE,
                 seed: int | None = RNG_SEED, weight_threshold: float = WEIGHT_THRESHOLD,
                 survival_chance: float = SURVIVAL_CHANCE):
        self.medium = medium
        self.device = torch.device(device)
        # Phase functions that hold precomputed tensors (e.g. Mie's tabulated
        # CDF/Mueller matrix) need moving onto the simulation's device/dtype;
        # HG/isotropic are pure scalar math and don't define `to()`.
        if hasattr(phase_function, "to"):
            phase_function = phase_function.to(self.device, DTYPE)
        self.phase_function = phase_function
        self.n_photons = n_photons
        self.weight_threshold = weight_threshold   # roulette trigger
        self.survival_chance = survival_chance      # roulette survival probability
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

    def _rand(self, n: int) -> torch.Tensor:
        # Clamp away from 0 so log(xi) in sample_step never hits -inf.
        return torch.rand(n, generator=self.rng, device=self.device, dtype=DTYPE).clamp_(1e-7, 1.0)

    # ----- CDF inversion samplers (batched) ----------------------------------

    def sample_step(self, n: int) -> torch.Tensor:
        """Step lengths from exponential CDF.  s = -l_scat * ln(xi)."""
        return -self.medium.l_scat * torch.log(self._rand(n))

    def sample_cos_theta(self, n: int) -> torch.Tensor:
        """Polar scattering cosines from the configured phase function's CDF."""
        return self.phase_function.sample_cos_theta(self._rand(n))

    def sample_azimuth(self, n: int) -> torch.Tensor:
        """Azimuthal angles from uniform CDF on [0, 2pi)."""
        return 2.0 * math.pi * self._rand(n)

    def mueller_matrix(self, cos_theta: torch.Tensor) -> torch.Tensor:
        """Mueller matrix for the current scattering event.

        Uses the phase function's own Mueller matrix when it defines one
        (e.g. Mie, where the polarization model and the sampled scattering
        angle come from the same physical particle); otherwise falls back
        to the shared Rayleigh approximation, since HG/isotropic aren't
        tied to a specific particle's exact phase matrix.
        """
        if hasattr(self.phase_function, "mueller_matrix"):
            return self.phase_function.mueller_matrix(cos_theta)
        return rayleigh_mueller_matrix(cos_theta)

    @staticmethod
    def _normalize_polarization(stokes: torch.Tensor) -> torch.Tensor:
        """Renormalize a batch of Stokes vectors to I = 1 after a Mueller-matrix
        event (scattering or a Fresnel boundary interaction) -- polarization
        state only, photon energy bookkeeping is `weights`. Scattered/reflected/
        transmitted intensity can pass through a genuine physical null (e.g.
        fully polarized light hitting a Rayleigh 90deg null), so after
        renormalizing, clamp Q/U/V back onto the physical bound
        sqrt(Q^2+U^2+V^2) <= I instead of letting a near-zero divide blow up.
        """
        stokes = stokes / stokes[:, :1].clamp_min(1e-6)
        pol_mag = torch.linalg.norm(stokes[:, 1:], dim=-1, keepdim=True).clamp_min(1e-12)
        shrink = torch.clamp(1.0 / pol_mag, max=1.0)
        return torch.cat([stokes[:, :1], stokes[:, 1:] * shrink], dim=-1)

    def _sample_scattering_angle_polarized(self, n: int, stokes: torch.Tensor, alive: torch.Tensor,
                                            max_rounds: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (cos_theta, phi) via acceptance-rejection on the true
        scattered intensity for the photon's current polarization state,
        instead of sample_cos_theta()'s unpolarized phase function plus a
        phi uniform in [0, 2pi).

        For fixed theta, the actual scattered intensity as a function of phi
        is I(theta,phi) = M11(theta) + M12(theta)*(cos(2phi)*Q + sin(2phi)*U)
        (the photon's Stokes vector is kept normalized to I=1, see run()).
        Candidates (theta, phi) are drawn from the same unpolarized proposal
        sample_cos_theta()/sample_azimuth() used when this is disabled, and
        accepted with probability I(theta,phi) / [M11(theta) + |M12(theta)|*p],
        where p = sqrt(Q^2+U^2) is the photon's current linear-polarization
        magnitude -- the exact (tightest possible) envelope over phi, so
        unpolarized photons (p=0) always accept on the first try and recover
        the same unpolarized sampling as when this is disabled.

        Any photon still unresolved after `max_rounds` (only possible for
        strongly polarized light at sharply dichroic angles) is forced to
        accept its last candidate, trading a small bias for a bounded loop.
        """
        cos_t = torch.zeros(n, device=self.device, dtype=DTYPE)
        phi = torch.zeros(n, device=self.device, dtype=DTYPE)
        pending = alive.clone()

        q_in, u_in = stokes[:, 1], stokes[:, 2]
        p_in = torch.sqrt(q_in * q_in + u_in * u_in)

        for _ in range(max_rounds):
            if not pending.any():
                break

            cand_cos_t = self.sample_cos_theta(n)
            cand_phi = self.sample_azimuth(n)

            m = self.mueller_matrix(cand_cos_t)
            m11, m12 = m[:, 0, 0], m[:, 0, 1]
            cos2p, sin2p = torch.cos(2.0 * cand_phi), torch.sin(2.0 * cand_phi)
            intensity = m11 + m12 * (cos2p * q_in + sin2p * u_in)
            bound = m11 + m12.abs() * p_in
            accept_prob = (intensity / bound.clamp_min(1e-12)).clamp(0.0, 1.0)
            accept = pending & (self._rand(n) < accept_prob)

            cos_t = torch.where(accept, cand_cos_t, cos_t)
            phi = torch.where(accept, cand_phi, phi)
            pending &= ~accept

        if pending.any():
            cos_t = torch.where(pending, cand_cos_t, cos_t)
            phi = torch.where(pending, cand_phi, phi)

        return cos_t, phi

    # ----- transport ----------------------------------------------------------

    def run(self, max_steps: int = MAX_STEPS, track_history: bool = False) -> dict[str, torch.Tensor]:
        """Launch and propagate the whole photon population together.

        If `track_history` is set, also records the position and alive mask
        after every step (returned as (T+1, n, 3) / (T+1, n) tensors,
        including the initial state at index 0). Only intended for small
        photon counts used for trajectory visualization -- recording this
        for the full simulated population would be far too much memory.
        """
        n = self.n_photons
        positions = torch.zeros(n, 3, device=self.device, dtype=DTYPE)
        directions = torch.zeros(n, 3, device=self.device, dtype=DTYPE)
        directions[:, 2] = 1.0
        weights = torch.ones(n, device=self.device, dtype=DTYPE)
        alive = torch.ones(n, dtype=torch.bool, device=self.device)

        position_history = [positions.clone()] if track_history else None
        alive_history = [alive.clone()] if track_history else None

        # Polarization reference frame: pol_ref is a unit vector perpendicular
        # to `direction` that defines the Stokes Q/U axes; it is parallel
        # transported (not re-derived from scratch) at every scattering event
        # so Q/U stay physically continuous between events. The x-axis is an
        # arbitrary but valid choice perpendicular to the launch direction.
        pol_ref = torch.zeros(n, 3, device=self.device, dtype=DTYPE)
        pol_ref[:, 0] = 1.0
        stokes = torch.tensor(INITIAL_STOKES, device=self.device, dtype=DTYPE).expand(n, 4).clone()
        stokes = stokes / stokes[:, :1].clamp_min(1e-12)  # normalize so I = 1

        # Which face an escaped photon exited through (True = z=0 "top", False = the
        # z=thickness "bottom"); meaningless for photons that are still alive or were
        # killed by roulette, but harmless there.
        exited_top = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Internal (pre-refraction) incidence angle cosine at the moment a photon
        # transmits through a face -- frozen there since a photon transmits at most
        # once -- kept separately from `direction` (which holds the *refracted* exit
        # direction after transmission) so both can be inspected/plotted.
        internal_incidence_cos = torch.zeros(n, device=self.device, dtype=DTYPE)
        z_hat = torch.zeros(n, 3, device=self.device, dtype=DTYPE)
        z_hat[:, 2] = 1.0

        remaining = n
        with tqdm(total=n, desc="Propagating photons", unit="photon") as pbar:
            for _ in range(max_steps):
                if remaining == 0:
                    break

                # ----- step, clipped at whichever slab face (if any) is reached first ---
                step = self.sample_step(n)
                dir_z = directions[:, 2]
                unclipped_z = positions[:, 2] + step * dir_z
                hits_top = alive & (unclipped_z < 0.0) & (dir_z < 0.0)
                hits_bottom = alive & (unclipped_z > self.medium.thickness) & (dir_z > 0.0)
                hits_boundary = hits_top | hits_bottom
                face_z = torch.where(hits_top, torch.zeros_like(unclipped_z),
                                      torch.full_like(unclipped_z, self.medium.thickness))
                safe_dir_z = torch.where(dir_z.abs() > 1e-8, dir_z, torch.full_like(dir_z, 1e-8))
                travelled = torch.where(hits_boundary, (face_z - positions[:, 2]) / safe_dir_z, step)

                positions = positions + (alive.unsqueeze(-1) * travelled.unsqueeze(-1)) * directions
                weights = torch.where(alive, weights * torch.exp(-self.medium.mu_a * travelled), weights)

                # ----- Fresnel reflection/transmission at a slab face --------------------
                # Weighted by the photon's actual (rotated-into-plane-of-incidence) Stokes
                # vector rather than a polarization-blind average of Rs/Rp. TIR falls out
                # of fresnel_mueller_matrices automatically (reflectance == 1 there).
                cos_i = dir_z.abs().clamp(1e-6, 1.0)
                transverse_raw = directions - dir_z.unsqueeze(-1) * z_hat
                # sin_i from the transverse vector's own norm, not sqrt(1 - cos_i^2): the
                # latter cancels catastrophically near normal incidence in float32 (cos_i
                # rounds to exactly 1.0 for any true angle within a fraction of a degree),
                # which would corrupt `transverse`'s magnitude and, with it, pol_ref.
                sin_i = torch.linalg.norm(transverse_raw, dim=-1)
                degenerate = (sin_i < 1e-6).unsqueeze(-1)  # direction ~parallel to the face normal
                transverse = transverse_raw / sin_i.clamp_min(1e-6).unsqueeze(-1)

                r_mueller, t_mueller, cos_refr = fresnel_mueller_matrices(cos_i, self.medium.n, self.medium.n_outside)

                # Rotate the carried Stokes vector so Q aligns with e_p (in-plane,
                # perpendicular to `direction` -- the "p" axis of the plane of incidence).
                # Falls back to the existing pol_ref at normal incidence, where e_p is
                # undefined but also irrelevant (Rs == Rp there, so any axis will do).
                e2_current = torch.linalg.cross(directions, pol_ref, dim=-1)
                e_p_raw = z_hat - dir_z.unsqueeze(-1) * directions
                e_p = torch.where(degenerate, pol_ref,
                                   e_p_raw / torch.linalg.norm(e_p_raw, dim=-1, keepdim=True).clamp_min(1e-8))
                phi_boundary = torch.atan2(torch.einsum("ni,ni->n", e_p, e2_current),
                                            torch.einsum("ni,ni->n", e_p, pol_ref))
                stokes_ps = rotate_stokes(stokes, phi_boundary)

                reflectance = torch.einsum("nj,nj->n", r_mueller[:, 0, :], stokes_ps).clamp(0.0, 1.0)
                reflect = hits_boundary & (self._rand(n) < reflectance)
                transmit = hits_boundary & ~reflect

                # Reflection: specular (mirror the face-normal component of direction and
                # of the polarization frame). U and V also flip sign -- reflection reverses
                # the local frame's handedness, the same physical effect that makes mirrors
                # reverse the handedness of circularly polarized light -- on top of whatever
                # the Fresnel reflection Mueller matrix itself does to the polarization state.
                flip_z = torch.tensor([1.0, 1.0, -1.0], device=self.device, dtype=DTYPE)
                reflected_dir = directions * flip_z
                reflected_pol_ref = e_p * flip_z
                reflected_stokes = self._normalize_polarization(torch.einsum("nij,nj->ni", r_mueller, stokes_ps))
                reflected_stokes = reflected_stokes * torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device, dtype=DTYPE)

                # Transmission (escape): refracted via Snell's law within the plane of
                # incidence -- the transverse (in-face) direction is preserved, only the
                # face-normal component changes with the new angle.
                sin_refr = ((self.medium.n / self.medium.n_outside) * sin_i).clamp(max=1.0)
                sign_z = torch.sign(dir_z)
                transmitted_dir = sin_refr.unsqueeze(-1) * transverse + (sign_z * cos_refr).unsqueeze(-1) * z_hat
                transmitted_pol_ref = torch.where(
                    degenerate, pol_ref,
                    cos_refr.unsqueeze(-1) * transverse - (sign_z * sin_refr).unsqueeze(-1) * z_hat,
                )
                transmitted_stokes = self._normalize_polarization(torch.einsum("nij,nj->ni", t_mueller, stokes_ps))

                reflect_3, transmit_3 = reflect.unsqueeze(-1), transmit.unsqueeze(-1)
                directions = torch.where(reflect_3, reflected_dir, torch.where(transmit_3, transmitted_dir, directions))
                pol_ref = torch.where(reflect_3, reflected_pol_ref, torch.where(transmit_3, transmitted_pol_ref, pol_ref))
                stokes = torch.where(reflect_3.expand(-1, 4), reflected_stokes,
                                      torch.where(transmit_3.expand(-1, 4), transmitted_stokes, stokes))
                exited_top = torch.where(transmit & hits_top, torch.ones_like(exited_top), exited_top)
                internal_incidence_cos = torch.where(transmit, cos_i, internal_incidence_cos)

                alive &= ~transmit

                # Reflected photons already changed direction at the boundary this step;
                # by the memorylessness of the exponential step distribution, resuming with
                # a fresh sampled step (and scattering event) next iteration is statistically
                # identical to continuing this same step past the reflection point, so they
                # skip the phase-function scatter below for this step.
                scatter_mask = alive & ~reflect
                scatter_mask_3 = scatter_mask.unsqueeze(-1)

                # ----- phase-function scatter, using the carried (direction, --------
                # pol_ref) frame so the polarization reference stays continuous -----
                if USE_POLARIZED_SAMPLING:
                    cos_t, phi = self._sample_scattering_angle_polarized(n, stokes, scatter_mask)
                else:
                    cos_t = self.sample_cos_theta(n)
                    phi = self.sample_azimuth(n)
                sin_t = torch.sqrt((1.0 - cos_t * cos_t).clamp_min(0.0))
                cos_p, sin_p = torch.cos(phi), torch.sin(phi)

                e2 = torch.linalg.cross(directions, pol_ref, dim=-1)
                e1_prime = cos_p.unsqueeze(-1) * pol_ref + sin_p.unsqueeze(-1) * e2

                new_dir = sin_t.unsqueeze(-1) * e1_prime + cos_t.unsqueeze(-1) * directions
                new_dir = new_dir / torch.linalg.norm(new_dir, dim=-1, keepdim=True)
                new_pol_ref = cos_t.unsqueeze(-1) * e1_prime - sin_t.unsqueeze(-1) * directions
                new_pol_ref = new_pol_ref / torch.linalg.norm(new_pol_ref, dim=-1, keepdim=True)

                directions = torch.where(scatter_mask_3, new_dir, directions)
                pol_ref = torch.where(scatter_mask_3, new_pol_ref, pol_ref)

                # ----- Stokes vector update: rotate into the scattering plane, apply the
                # Mueller matrix, then renormalize to I = 1 (see _normalize_polarization).
                rotated = rotate_stokes(stokes, phi)
                scattered = self._normalize_polarization(torch.einsum("nij,nj->ni", self.mueller_matrix(cos_t), rotated))
                stokes = torch.where(scatter_mask_3.expand(-1, 4), scattered, stokes)

                # ----- Russian roulette, unbiased -----------------------------------
                below = alive & (weights < self.weight_threshold)
                survives = self._rand(n) < self.survival_chance
                killed = below & ~survives
                boosted = below & survives
                weights = torch.where(boosted, weights / self.survival_chance, weights)
                weights = torch.where(killed, torch.zeros_like(weights), weights)
                alive &= ~killed

                if track_history:
                    position_history.append(positions.clone())
                    alive_history.append(alive.clone())

                new_remaining = int(alive.sum().item())
                pbar.update(remaining - new_remaining)
                remaining = new_remaining

        result = {
            "position": positions, "direction": directions, "weight": weights,
            "alive": alive, "stokes": stokes, "pol_ref": pol_ref, "exited_top": exited_top,
            "internal_incidence_cos": internal_incidence_cos,
        }
        if track_history:
            result["position_history"] = torch.stack(position_history)  # (T+1, n, 3)
            result["alive_history"] = torch.stack(alive_history)        # (T+1, n)
        return result
