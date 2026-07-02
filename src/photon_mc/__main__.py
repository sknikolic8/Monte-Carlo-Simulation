"""CLI entry point: run the configured simulation and display the results.

    python -m photon_mc
"""

from __future__ import annotations

import torch

from . import params
from .medium import Medium
from .phase_functions import get_phase_function
from .plotting import (
    exit_angles,
    plot_backscatter_intensity,
    plot_backscatter_polarization,
    plot_fresnel_reflectance,
    plot_polar_azimuthal_scattering_probability,
    plot_polar_scattering_probability,
    plot_polarization_heatmap,
    plot_polarization_type_heatmaps,
    plot_trajectories,
    show_all_figures_tabbed,
)
from .simulation import MonteCarlo


def main() -> None:
    medium = Medium(mu_a=params.MU_A, mu_s=params.MU_S, n=params.N_MEDIUM,
                     n_outside=params.N_OUTSIDE, thickness=params.SLAB_THICKNESS)
    phase_function = get_phase_function(params.PHASE_FUNCTION, **params.PHASE_FUNCTION_PARAMS)
    mc = MonteCarlo(medium, phase_function, n_photons=params.N_PHOTONS, device=params.DEVICE, seed=params.RNG_SEED)
    print(f"Running on device: {mc.device}")

    plot_polar_scattering_probability(mc)
    plot_polar_azimuthal_scattering_probability(mc)
    plot_fresnel_reflectance(medium)

    result = mc.run(max_steps=params.MAX_STEPS)
    alive, weight, direction, position = result["alive"], result["weight"], result["direction"], result["position"]
    stokes, exited_top = result["stokes"], result["exited_top"]
    internal_incidence_cos = result["internal_incidence_cos"]

    # "Escaped" now splits into two faces: diffuse reflectance (back out z=0, the
    # illuminated top) and diffuse transmittance (through z=thickness, the bottom).
    top_mask = (~alive) & (weight > 0) & exited_top
    bottom_mask = (~alive) & (weight > 0) & ~exited_top
    n_escaped_top = int(top_mask.sum().item())
    n_escaped_bottom = int(bottom_mask.sum().item())
    diffuse_reflectance = (weight[top_mask].sum() / params.N_PHOTONS).item()
    diffuse_transmittance = (weight[bottom_mask].sum() / params.N_PHOTONS).item()

    print(f"Medium: mu_a={medium.mu_a}, mu_s={medium.mu_s}, n={medium.n}, n_outside={medium.n_outside}, "
          f"thickness={medium.thickness}")
    print(f"Phase function: {params.PHASE_FUNCTION} ({params.PHASE_FUNCTION_PARAMS})")
    print(f"Initial Stokes vector: {params.INITIAL_STOKES}")
    print(f"Scattering MFP = {medium.l_scat:.3f}, absorption length = {medium.l_abs:.3f}")
    print(f"Photons launched: {params.N_PHOTONS}")
    print(f"Photons escaped through top (z=0): {n_escaped_top}")
    print(f"Photons escaped through bottom (z={medium.thickness}): {n_escaped_bottom}")
    print(f"Estimated diffuse reflectance: {diffuse_reflectance:.4f}")
    print(f"Estimated diffuse transmittance: {diffuse_transmittance:.4f}")

    # Duplicate the escape-analysis plots for both faces, clearly labeled: the top
    # face (z=0) carries the diffuse *reflectance* signal, the bottom face
    # (z=thickness) the diffuse *transmittance* signal through the finite slab.
    for face_mask, face_label, outward_normal_z in (
        (top_mask, "top face (reflectance)", -1.0),
        (bottom_mask, "bottom face (transmittance)", 1.0),
    ):
        face_weight = weight[face_mask]

        # Each individual photon's Stokes vector stays exactly fully polarized
        # under a single Rayleigh-type Mueller matrix, so per-photon DOLP is
        # uninformative. What depolarizes is the *ensemble*: different photons
        # arrive with differently-oriented polarization after their own random
        # scattering history, so average their energy-weighted Stokes vectors
        # (as a detector would) before computing degree of polarization.
        mean_stokes = (stokes[face_mask] * face_weight.unsqueeze(-1)).sum(0) / face_weight.sum().clamp_min(1e-12)
        dolp = torch.sqrt(mean_stokes[1] ** 2 + mean_stokes[2] ** 2) / mean_stokes[0].clamp_min(1e-12)
        docp = mean_stokes[3].abs() / mean_stokes[0].clamp_min(1e-12)
        print(f"Ensemble degree of linear polarization ({face_label}): {dolp.item():.4f}")
        print(f"Ensemble degree of circular polarization ({face_label}): {docp.item():.4f}")

        # External (post-refraction, in the surrounding medium -- unbounded by the
        # critical angle) and internal (pre-refraction, at the boundary -- bounded
        # by the critical angle) exit angles are genuinely different quantities;
        # plot both, clearly labeled, rather than conflating them (see exit_angles
        # and MonteCarlo.run()'s `internal_incidence_cos`).
        external_angles, _ = exit_angles(direction[face_mask], face_weight, outward_normal_z)
        internal_angles = torch.arccos(torch.clamp(internal_incidence_cos[face_mask], -1.0, 1.0))
        for angle_kind, angles in (("external", external_angles), ("internal", internal_angles)):
            plot_backscatter_intensity(angles, face_weight, n_photons=params.N_PHOTONS, n_bins=params.N_ANGLE_BINS,
                                        label=face_label, angle_kind=angle_kind)
            plot_backscatter_polarization(angles, face_weight, stokes[face_mask], n_bins=params.N_ANGLE_BINS,
                                           label=face_label, angle_kind=angle_kind)
        plot_polarization_heatmap(position[face_mask], face_weight, stokes[face_mask],
                                   n_bins=params.N_SPATIAL_BINS, label=face_label)
        plot_polarization_type_heatmaps(position[face_mask], face_weight, stokes[face_mask],
                                         initial_stokes=params.INITIAL_STOKES, n_bins=params.N_SPATIAL_BINS,
                                         label=face_label)

    # Dedicated small batch with full position history recorded, just for
    # trajectory visualization -- doing this for all N_PHOTONS would need
    # far too much memory.
    traj_mc = MonteCarlo(medium, phase_function, n_photons=params.N_TRAJECTORY_PHOTONS, device=params.DEVICE,
                          seed=params.RNG_SEED + 1)
    traj_result = traj_mc.run(max_steps=params.MAX_STEPS, track_history=True)
    plot_trajectories(traj_result["position_history"], traj_result["alive_history"], traj_result["weight"],
                       thickness=medium.thickness, n_trajectories=params.N_TRAJECTORIES_TO_PLOT)

    show_all_figures_tabbed()


if __name__ == "__main__":
    main()
