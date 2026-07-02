"""Analysis and visualization for MonteCarlo simulation results.

The plot_* functions below build ordinary headless matplotlib Figures;
`show_all_figures_tabbed` is the one place that actually renders them, in a
single Tk window switched via a dropdown menu instead of one OS window per
figure.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # figures are built headless; show_all_figures_tabbed() displays them all in one window
import matplotlib.pyplot as plt
import torch

from .params import (
    AZIMUTHAL_PLOT_THETA_DEG, DTYPE, INITIAL_STOKES, N_ANGLE_BINS, N_SPATIAL_BINS, N_TRAJECTORIES_TO_PLOT, RNG_SEED,
)
from .phase_functions.mueller import fresnel_mueller_matrices

if TYPE_CHECKING:
    from .medium import Medium
    from .simulation import MonteCarlo


def exit_angles(escaped_directions: torch.Tensor, escaped_weights: torch.Tensor,
                 outward_normal_z: float) -> tuple[torch.Tensor, torch.Tensor]:
    """*External* exit polar angle (radians, measured from the face's outward
    normal, in the surrounding medium) for each photon that escaped through
    that face -- i.e. the refracted angle after Snell's law bends the photon
    at the boundary, not the internal angle it struck the face at (see
    MonteCarlo.run()'s `internal_incidence_cos`, which is bounded by the
    critical angle; this external angle is not, since refraction fans the
    internal escape cone across the full external hemisphere).

    `outward_normal_z` is -1.0 for the top face (z=0, normal points in -z)
    or +1.0 for the bottom face (z=thickness, normal points in +z).

    A photon's `direction` is left untouched once it escapes (it stops
    being updated once `alive` goes False), so it still holds the
    trajectory the photon was exiting on.
    """
    cos_theta = torch.clamp(outward_normal_z * escaped_directions[:, 2], -1.0, 1.0)
    angles = torch.arccos(cos_theta)
    return angles, escaped_weights


def _bin_backscatter_intensity(angles: torch.Tensor, weights: torch.Tensor, n_photons: int,
                                n_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin escaped photons by backscatter angle.

    Returns (bin center angle [deg], intensity [per launched photon per
    steradian]) -- the shared computation behind both backscatter-intensity
    plots below.
    """
    angles = angles.detach().to("cpu", torch.float64)
    weights = weights.detach().to("cpu", torch.float64)

    bin_edges = torch.linspace(0.0, math.pi / 2, n_bins + 1, dtype=torch.float64)
    bin_idx = torch.clamp(torch.bucketize(angles, bin_edges, right=True) - 1, 0, n_bins - 1)
    weight_sum = torch.zeros(n_bins, dtype=torch.float64).scatter_add_(0, bin_idx, weights)

    # Solid angle of each polar ring, so equal-angle bins near grazing
    # incidence (which cover more solid angle) aren't over-weighted.
    solid_angle = 2.0 * math.pi * (torch.cos(bin_edges[:-1]) - torch.cos(bin_edges[1:]))
    intensity = weight_sum / (n_photons * solid_angle)
    bin_centers_deg = torch.rad2deg(0.5 * (bin_edges[:-1] + bin_edges[1:]))
    return bin_centers_deg, intensity


def _angle_axis_label(angle_kind: str) -> str:
    """x-axis label for the two `angle_kind` variants plotted by
    plot_backscatter_intensity/plot_backscatter_polarization below."""
    return ("Internal exit angle (pre-refraction, at the boundary) [deg]" if angle_kind == "internal"
            else "External exit angle (post-refraction, in the surrounding medium) [deg]")


def plot_polar_scattering_probability(mc: "MonteCarlo", n_points: int = 721) -> None:
    """Polar plot of the (unpolarized) phase function's scattering
    probability vs. polar angle theta.

    Plots M11(theta) (the same quantity sample_cos_theta() draws from) for
    theta in [0, pi], mirrored across the forward axis to produce the
    familiar symmetric lobe shape of a scattering diagram, rather than just
    a half-circle wedge.
    """
    theta = torch.linspace(0.0, math.pi, n_points, device=mc.device, dtype=DTYPE)
    m11 = mc.mueller_matrix(torch.cos(theta))[:, 0, 0].detach().to("cpu", torch.float64)
    theta_cpu = theta.detach().to("cpu", torch.float64)

    full_theta = torch.cat([theta_cpu, 2.0 * math.pi - theta_cpu.flip(0)])
    full_m11 = torch.cat([m11, m11.flip(0)])

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.plot(full_theta.numpy(), full_m11.numpy())
    ax.set_theta_zero_location("N")  # forward scattering (theta=0) points "up"
    ax.set_title("Scattering probability vs. polar angle\n(0deg = forward, 180deg = backward)")
    fig.tight_layout()


def plot_fresnel_reflectance(medium: "Medium", n_points: int = 721) -> None:
    """Plot the Fresnel reflectance of a slab face (Rs, Rp, and their
    unpolarized average) vs. angle of incidence, for the configured medium
    -> outside index pair -- a direct visual check of the boundary physics
    used in MonteCarlo.run(), including the total-internal-reflection
    cutoff (vertical line) when medium.n > medium.n_outside: reflectance
    should jump to exactly 1 (transmittance to exactly 0) right at the
    critical angle and stay there for every angle beyond it.
    """
    theta_deg = torch.linspace(0.0, 90.0, n_points, dtype=torch.float64)
    cos_i = torch.cos(torch.deg2rad(theta_deg))
    R, _, _ = fresnel_mueller_matrices(cos_i, medium.n, medium.n_outside)
    r00, r01 = R[:, 0, 0], R[:, 0, 1]  # row0 = [0.5*(Rp+Rs), 0.5*(Rp-Rs), 0, 0]
    Rs, Rp = r00 - r01, r00 + r01

    fig, ax = plt.subplots()
    ax.plot(theta_deg.numpy(), Rs.numpy(), label="Rs (s-polarized / perpendicular)")
    ax.plot(theta_deg.numpy(), Rp.numpy(), label="Rp (p-polarized / parallel)")
    ax.plot(theta_deg.numpy(), r00.numpy(), "--", color="gray", label="unpolarized average")
    if medium.n > medium.n_outside:
        theta_c = math.degrees(math.asin(medium.n_outside / medium.n))
        ax.axvline(theta_c, color="red", linestyle=":", label=f"critical angle = {theta_c:.1f}deg")
    ax.set_xlabel("Angle of incidence [deg]")
    ax.set_ylabel("Reflectance")
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Fresnel reflectance at a slab face (n={medium.n} -> n_outside={medium.n_outside})")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()


def _azimuthal_scattering_intensity(mc: "MonteCarlo", theta_deg: float, q_in: float, u_in: float,
                                     n_points: int = 721) -> tuple[torch.Tensor, torch.Tensor]:
    """Scattering intensity I(phi) = M11 + M12*(cos(2phi)*Q + sin(2phi)*U) at
    a fixed polar angle, for incident linear-polarization components (Q, U).

    Shared by the single- and multi-panel azimuthal scattering-probability
    plots below. Note V (circular polarization) never enters this formula:
    under this block-diagonal Mueller matrix structure, circular
    polarization has no effect on the azimuthal scattering pattern.
    """
    cos_t = torch.tensor([math.cos(math.radians(theta_deg))], device=mc.device, dtype=DTYPE)
    m = mc.mueller_matrix(cos_t)
    m11, m12 = m[0, 0, 0].item(), m[0, 0, 1].item()

    phi = torch.linspace(0.0, 2.0 * math.pi, n_points, dtype=torch.float64)
    intensity = m11 + m12 * (torch.cos(2.0 * phi) * q_in + torch.sin(2.0 * phi) * u_in)
    return phi, intensity


def _draw_polarization_direction_line(ax, q_in: float, u_in: float) -> None:
    """Draw a diameter line across a polar axis marking the incident
    linear-polarization orientation psi = 0.5*atan2(U, Q); skipped if there
    is no linear-polarization component (e.g. unpolarized or circular)."""
    if math.hypot(q_in, u_in) <= 1e-6:
        return
    psi = 0.5 * math.atan2(u_in, q_in)
    r_edge = ax.get_rmax()
    ax.plot([psi, psi + math.pi], [r_edge, r_edge], color="red", linewidth=2,
            label="linear polarization direction")


def plot_polar_azimuthal_scattering_probability(mc: "MonteCarlo", stokes: tuple = INITIAL_STOKES,
                                                 theta_deg: float = AZIMUTHAL_PLOT_THETA_DEG,
                                                 n_points: int = 721) -> None:
    """Polar plot of scattering probability vs. azimuthal angle phi at a
    fixed polar angle, for the given incident Stokes vector.

    Unpolarized light scatters uniformly in phi (a plain circle); linear
    polarization (Q, U) introduces the characteristic two-lobed azimuthal
    dependence. A red line marks the incident linear-polarization direction
    (skipped if there's no linear polarization component to show).
    """
    q_in, u_in = stokes[1], stokes[2]
    phi, intensity = _azimuthal_scattering_intensity(mc, theta_deg, q_in, u_in, n_points)

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.plot(phi.numpy(), intensity.numpy())
    _draw_polarization_direction_line(ax, q_in, u_in)
    if math.hypot(q_in, u_in) > 1e-6:
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    ax.set_title(f"Scattering probability vs. azimuthal angle\n(polar angle = {theta_deg:.0f}deg)")
    fig.tight_layout()


def plot_backscatter_intensity(angles: torch.Tensor, weights: torch.Tensor, n_photons: int,
                                n_bins: int = N_ANGLE_BINS, label: str = "",
                                angle_kind: str = "external") -> None:
    """Plot diffusely escaped intensity (per launched photon, per steradian)
    as a function of exit angle from the face normal. `label` (e.g. "top
    face (reflectance)") is appended to the title to distinguish which face
    this is for. `angle_kind` ("external" or "internal") selects whether
    `angles` is the refracted angle in the surrounding medium (unbounded by
    the critical angle) or the pre-refraction angle of incidence at the
    boundary (bounded by the critical angle) -- see `exit_angles` and
    MonteCarlo.run()'s `internal_incidence_cos`."""
    bin_centers_deg, intensity = _bin_backscatter_intensity(angles, weights, n_photons, n_bins)

    fig, ax = plt.subplots()
    ax.plot(bin_centers_deg.numpy(), intensity.numpy(), marker="o")
    ax.set_xlabel(_angle_axis_label(angle_kind))
    ax.set_ylabel("Intensity [per launched photon per steradian]")
    title = f"Diffuse exit intensity vs. {angle_kind} exit angle"
    ax.set_title(f"{title} -- {label}" if label else title)
    ax.grid(True)
    fig.tight_layout()


def plot_backscatter_polarization(angles: torch.Tensor, weights: torch.Tensor, stokes: torch.Tensor,
                                   n_bins: int = N_ANGLE_BINS, label: str = "",
                                   angle_kind: str = "external") -> None:
    """Plot the ensemble degree of linear/circular polarization of escaped
    photons as a function of exit angle from the face normal. `label` (e.g.
    "bottom face (transmittance)") is appended to the title. `angle_kind`
    ("external" or "internal") selects which angle `angles` represents --
    see plot_backscatter_intensity.

    A single photon's Stokes vector stays exactly fully polarized under one
    Rayleigh Mueller-matrix event (see MonteCarlo.run()), so degree of
    polarization is only meaningful as an energy-weighted ensemble average
    within each angular bin -- what a polarization-sensitive detector at
    that angle would actually measure.
    """
    angles = angles.detach().to("cpu", torch.float64)
    weights = weights.detach().to("cpu", torch.float64)
    stokes = stokes.detach().to("cpu", torch.float64)

    bin_edges = torch.linspace(0.0, math.pi / 2, n_bins + 1, dtype=torch.float64)
    bin_idx = torch.clamp(torch.bucketize(angles, bin_edges, right=True) - 1, 0, n_bins - 1)

    weight_sum = torch.zeros(n_bins, dtype=torch.float64).scatter_add_(0, bin_idx, weights)
    stokes_sum = torch.zeros(n_bins, 4, dtype=torch.float64).scatter_add_(
        0, bin_idx.unsqueeze(-1).expand(-1, 4), stokes * weights.unsqueeze(-1)
    )
    mean_stokes = stokes_sum / weight_sum.unsqueeze(-1).clamp_min(1e-12)

    dolp = torch.sqrt(mean_stokes[:, 1] ** 2 + mean_stokes[:, 2] ** 2) / mean_stokes[:, 0].clamp_min(1e-12)
    docp = mean_stokes[:, 3].abs() / mean_stokes[:, 0].clamp_min(1e-12)
    bin_centers_deg = torch.rad2deg(0.5 * (bin_edges[:-1] + bin_edges[1:]))

    fig, ax = plt.subplots()
    ax.plot(bin_centers_deg.numpy(), dolp.numpy(), marker="o", label="degree of linear polarization")
    ax.plot(bin_centers_deg.numpy(), docp.numpy(), marker="s", label="degree of circular polarization")
    ax.set_xlabel(_angle_axis_label(angle_kind))
    ax.set_ylabel("Ensemble degree of polarization")
    title = f"Diffuse exit polarization vs. {angle_kind} exit angle"
    ax.set_title(f"{title} -- {label}" if label else title)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()


def _bin_stokes_by_position(position: torch.Tensor, weight: torch.Tensor, stokes: torch.Tensor,
                             n_bins: int) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Bin escaped photons by exit position (x, y) on the surface (z = 0)
    and compute the energy-weighted ensemble Stokes vector in each bin.

    As with the angle-resolved polarization plots, degree of polarization is
    only meaningful as an energy-weighted ensemble average within a bin --
    what a spatially resolved polarization-sensitive detector at the
    surface would measure -- not a per-photon quantity. Shared by
    plot_polarization_heatmap and plot_polarization_type_heatmaps.

    Returns (extent, mean_stokes [n_bins, n_bins, 4], weight_sum [n_bins, n_bins]).
    """
    x = position[:, 0].detach().to("cpu", torch.float64)
    y = position[:, 1].detach().to("cpu", torch.float64)
    weight = weight.detach().to("cpu", torch.float64)
    stokes = stokes.detach().to("cpu", torch.float64)

    # Robust symmetric extent: most photons stay within a few scattering
    # mean free paths of the source, but a handful travel much farther.
    extent = max(torch.quantile(torch.sqrt(x * x + y * y), 0.99).item(), 1e-6)

    edges = torch.linspace(-extent, extent, n_bins + 1, dtype=torch.float64)
    ix = torch.clamp(torch.bucketize(x, edges, right=True) - 1, 0, n_bins - 1)
    iy = torch.clamp(torch.bucketize(y, edges, right=True) - 1, 0, n_bins - 1)
    flat_idx = ix * n_bins + iy

    weight_sum = torch.zeros(n_bins * n_bins, dtype=torch.float64).scatter_add_(0, flat_idx, weight)
    stokes_sum = torch.zeros(n_bins * n_bins, 4, dtype=torch.float64).scatter_add_(
        0, flat_idx.unsqueeze(-1).expand(-1, 4), stokes * weight.unsqueeze(-1)
    )
    mean_stokes = stokes_sum / weight_sum.clamp_min(1e-12).unsqueeze(-1)
    return extent, mean_stokes.reshape(n_bins, n_bins, 4), weight_sum.reshape(n_bins, n_bins)


def plot_polarization_heatmap(position: torch.Tensor, weight: torch.Tensor, stokes: torch.Tensor,
                               n_bins: int = N_SPATIAL_BINS, label: str = "") -> None:
    """2D heatmap of the ensemble degree of linear polarization at
    emergence, binned by exit position (x, y) on the face. `label` (e.g.
    "top face (reflectance)") is appended to the title."""
    extent, mean_stokes, weight_sum = _bin_stokes_by_position(position, weight, stokes, n_bins)

    dolp = torch.sqrt(mean_stokes[..., 1] ** 2 + mean_stokes[..., 2] ** 2) / mean_stokes[..., 0].clamp_min(1e-12)
    dolp = torch.where(weight_sum > 0, dolp, torch.full_like(dolp, float("nan")))  # blank empty bins
    dolp_grid = dolp.T  # rows -> y, cols -> x, to match imshow's axis convention

    fig, ax = plt.subplots()
    im = ax.imshow(dolp_grid.numpy(), origin="lower", extent=[-extent, extent, -extent, extent],
                    cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Degree of linear polarization")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Degree of linear polarization at emergence -- {label}" if label
                 else "Degree of linear polarization at emergence")
    fig.tight_layout()


def plot_polarization_type_heatmaps(position: torch.Tensor, weight: torch.Tensor, stokes: torch.Tensor,
                                     initial_stokes: tuple = INITIAL_STOKES,
                                     n_bins: int = N_SPATIAL_BINS, label: str = "") -> None:
    """Side-by-side spatial heatmaps of the exiting (escaped) photons'
    polarization, decomposed relative to the incident launch direction
    (every photon is launched with the same INITIAL_STOKES):

      - Linear (co-polarized): fraction of exit intensity that would pass a
        linear polarizer aligned with the incident polarization direction.
      - Linear (orthogonal): fraction that would pass a polarizer rotated
        90deg from the incident direction (the classic cross-polarized
        channel used in polarization-difference imaging).
      - Circular: degree of circular polarization (|V|/I) at exit.
      - Total degree of polarization: sqrt(Q^2+U^2+V^2)/I at exit -- the
        orientation-agnostic "how polarized is the light here at all",
        combining linear and circular into one bounded quantity.

    Uses Malus's law in Stokes form: intensity through a linear polarizer at
    orientation psi is 0.5*(I + Q*cos(2psi) + U*sin(2psi)); psi0 is fixed by
    INITIAL_STOKES since all photons share the same launch polarization.

    All four panels share one color scale, fixed to the *incident* degree of
    polarization (not autoscaled to whatever the exit data happens to reach)
    -- a passively scattering medium can only preserve or reduce net
    polarization on average, never manufacture more of it than was launched,
    so the input DoP is the natural, run-independent ceiling to compare
    against.
    """
    extent, mean_stokes, weight_sum = _bin_stokes_by_position(position, weight, stokes, n_bins)
    mask = weight_sum > 0

    i0, q0, u0, v0 = initial_stokes
    p0 = math.hypot(q0, u0)
    # If launched unpolarized (or purely circular), there's no natural linear
    # reference axis; fall back to an arbitrary one (co/cross-pol are then
    # equal by symmetry, so the choice doesn't matter).
    cos2psi0, sin2psi0 = (q0 / p0, u0 / p0) if p0 > 1e-6 else (1.0, 0.0)
    dop_in = math.sqrt(q0 * q0 + u0 * u0 + v0 * v0) / max(i0, 1e-12)

    i_total = mean_stokes[..., 0].clamp_min(1e-12)
    proj = (mean_stokes[..., 1] * cos2psi0 + mean_stokes[..., 2] * sin2psi0) / i_total
    co_polarized = 0.5 * (1.0 + proj)
    cross_polarized = 0.5 * (1.0 - proj)
    circular = mean_stokes[..., 3].abs() / i_total
    total_dop = torch.sqrt(mean_stokes[..., 1] ** 2 + mean_stokes[..., 2] ** 2 + mean_stokes[..., 3] ** 2) / i_total

    panels = [
        ("Linear (co-polarized)", co_polarized),
        ("Linear (orthogonal)", cross_polarized),
        ("Circular", circular),
        ("Total degree of polarization", total_dop),
    ]

    # Shared color scale across all four panels, fixed to the incident DoP
    # (see docstring) rather than autoscaled to the exit data's own max.
    vmax = dop_in if dop_in > 1e-6 else 1.0

    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    im = None
    for ax, (title, grid) in zip(axes, panels):
        grid = torch.where(mask, grid, torch.full_like(grid, float("nan"))).T
        im = ax.imshow(grid.numpy(), origin="lower", extent=[-extent, extent, -extent, extent],
                        cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(title)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Fraction / degree of polarization")
    title_prefix = f"Exit polarization by type -- {label}" if label else "Exit polarization by type"
    fig.suptitle(f"{title_prefix}, relative to incident polarization direction "
                 f"(color scale fixed to incident DoP = {dop_in:.3f})")


def plot_trajectories(position_history: torch.Tensor, alive_history: torch.Tensor, weight: torch.Tensor,
                       thickness: float, n_trajectories: int = N_TRAJECTORIES_TO_PLOT, seed: int = RNG_SEED) -> None:
    """3D plot of a handful of representative photon trajectories, chosen
    from photons in `position_history`'s batch that escaped through either
    slab face (top z=0 or bottom z=thickness) with nonzero weight -- a
    trajectory that reflects off a face partway through still continues, so
    only the final exit (if any) counts as escaped.

    Each photon's trajectory is trimmed at the step where it stopped being
    alive (its exit point), since position/alive freeze from then on.
    """
    alive_final = alive_history[-1]
    escaped = (~alive_final) & (weight > 0)
    candidates = torch.nonzero(escaped, as_tuple=False).flatten()
    if candidates.numel() == 0:
        print("No escaped photons in the trajectory sample; skipping trajectory plot.")
        return

    rng = torch.Generator().manual_seed(seed)
    n_pick = min(n_trajectories, candidates.numel())
    pick = candidates[torch.randperm(candidates.numel(), generator=rng)[:n_pick]]

    position_history = position_history.detach().cpu()
    alive_history = alive_history.detach().cpu()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for idx in pick.tolist():
        not_alive = ~alive_history[:, idx]
        end = int(torch.argmax(not_alive.int()).item())  # first step photon stopped being alive
        path = position_history[: end + 1, idx].numpy()
        ax.plot(path[:, 0], path[:, 1], path[:, 2], marker=".", markersize=2, linewidth=1)
        ax.scatter(*path[0], color="green", s=15)   # launch point
        ax.scatter(*path[-1], color="red", s=15)    # exit point

    # Translucent planes marking the two slab boundaries, z = 0 and z = thickness.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx, yy = torch.meshgrid(torch.linspace(*xlim, 2), torch.linspace(*ylim, 2), indexing="ij")
    for face_z in (0.0, thickness):
        ax.plot_surface(xx.numpy(), yy.numpy(), torch.full_like(xx, face_z).numpy(), alpha=0.15, color="gray")

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(f"{n_pick} representative escaped photon trajectories")


def _figure_label(fig: "plt.Figure", num: int, max_len: int = 40) -> str:
    """Short tab label for a figure: its suptitle, else its first axes'
    title, else a generic fallback -- truncated so the tab strip stays tidy."""
    title = fig.get_suptitle() or next((ax.get_title() for ax in fig.axes if ax.get_title()), "")
    title = title or f"Figure {num}"
    return title if len(title) <= max_len else title[: max_len - 3] + "..."


def show_all_figures_tabbed() -> None:
    """Display every currently open matplotlib figure in a single Tk window,
    switched via a dropdown menu instead of one OS window per figure. Purely
    a display-layer change: the plot_* functions above still just build
    ordinary matplotlib Figures via pyplot (headless, since
    matplotlib.use("Agg") is set at import time); this is the one place that
    actually renders them.
    """
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    fig_nums = plt.get_fignums()
    if not fig_nums:
        return
    labels = [_figure_label(plt.figure(num), num, max_len=80) for num in fig_nums]

    root = tk.Tk()
    root.title("Monte Carlo photon transport -- results")
    root.geometry("1280x900")

    selector = ttk.Combobox(root, values=labels, state="readonly")
    selector.pack(fill="x", padx=4, pady=4)

    plot_area = ttk.Frame(root)
    plot_area.pack(fill="both", expand=True)
    displayed_widgets: list[tk.Widget] = []

    def show_figure(index: int) -> None:
        for widget in displayed_widgets:
            widget.destroy()
        displayed_widgets.clear()

        fig = plt.figure(fig_nums[index])
        canvas = FigureCanvasTkAgg(fig, master=plot_area)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, plot_area)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        displayed_widgets.extend([canvas.get_tk_widget(), toolbar])

    selector.bind("<<ComboboxSelected>>", lambda event: show_figure(selector.current()))
    selector.current(0)
    show_figure(0)

    root.mainloop()
