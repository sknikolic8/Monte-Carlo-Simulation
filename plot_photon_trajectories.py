#!/usr/bin/env python3
"""
Plot sample 2D photon trajectories from a Monte Carlo tissue optics simulation.

Expected real trajectory format:
    trajectories = [
        np.array([[x0, z0], [x1, z1], [x2, z2], ...]),
        np.array([[x0, z0], [x1, z1], [x2, z2], ...]),
        ...
    ]

Each photon path can have many scattering events.
This script plots only a small subset of photons and only the first N events.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#%%

# ============================================================
# MOCK DATA GENERATOR
# Replace this later with real simulation output
# ============================================================

def generate_mock_photon_paths(
    n_photons=5000,
    max_events=30,
    step_size_mean=0.12,
    seed=7,
):
    """
    Generate fake 2D photon paths for plotting development.

    x = lateral position in mm
    z = depth in mm
    source starts at (0, 0)
    """
    rng = np.random.default_rng(seed)
    trajectories = []

    for _ in range(n_photons):
        x, z = 0.0, 0.0
        path = [[x, z]]

        # Initial direction mostly downward into tissue
        angle = rng.normal(loc=np.pi / 2, scale=0.25)

        for _ in range(max_events):
            step = rng.exponential(step_size_mean)

            # Random scattering angle update
            angle += rng.normal(loc=0.0, scale=0.65)

            dx = step * np.cos(angle)
            dz = step * np.sin(angle)

            x += dx
            z += abs(dz)  # keeps mock photons mostly moving into tissue

            path.append([x, z])

            # Optional stopping condition if photon goes too deep
            if z > 2.0:
                break

        trajectories.append(np.array(path))

    return trajectories
#%%
# ============================================================
# PLOTTING FUNCTION
# ============================================================

def plot_photon_trajectories_2d(
    trajectories,
    n_photons_to_plot=10,
    max_scattering_events=10,
    total_photons=None,
    mu_a=0.01,
    mu_s=1.0,
    g=0.90,
    n_tissue=1.4,
    tissue_depth_mm=2.0,
    xlim=(-1.0, 1.0),
    save_path=None,
):
    """
    Plot a small number of photon trajectories in 2D.

    Parameters
    ----------
    trajectories : list of np.ndarray
        Each element should be shape (n_events, 2), columns [x, z].
    n_photons_to_plot : int
        Number of photon paths to display.
    max_scattering_events : int
        Number of scattering events to show per photon.
    total_photons : int or None
        Total photons in simulation. If None, uses len(trajectories).
    """

    if total_photons is None:
        total_photons = len(trajectories)

    n_available = len(trajectories)
    n_plot = min(n_photons_to_plot, n_available)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Air-tissue boundary
    ax.axhline(0, linewidth=1.5, color="black")
    ax.fill_between(
        [xlim[0], xlim[1]],
        -0.22,
        0,
        alpha=0.15,
        hatch="///",
        edgecolor="gray",
        facecolor="none",
    )

    # Plot selected photon paths
    for i in range(n_plot):
        path = trajectories[i]

        # Keep source + first max_scattering_events
        path_to_plot = path[: max_scattering_events + 1]

        x = path_to_plot[:, 0]
        z = path_to_plot[:, 1]

        ax.plot(
            x,
            z,
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=f"Photon {i + 1}",
        )

    # Source marker
    ax.scatter(
        0,
        0,
        marker="*",
        s=250,
        color="red",
        edgecolor="black",
        zorder=10,
        label="Source (x=0, z=0)",
    )

    # Labels for regions
    ax.text(xlim[1] - 0.05, -0.08, "Air", ha="right", va="center")
    ax.text(xlim[1] - 0.05, 0.10, f"Tissue (n={n_tissue})", ha="right", va="center")

    # Simulation info box
    info_text = (
        "Simulation Info\n"
        f"Total photons simulated: {total_photons}\n"
        f"Photons displayed: {n_plot}\n"
        f"Max scattering events shown: {max_scattering_events}\n\n"
        f"$\\mu_a$ = {mu_a} mm$^{{-1}}$\n"
        f"$\\mu_s$ = {mu_s} mm$^{{-1}}$\n"
        f"$g$ = {g}\n"
        f"$n$ = {n_tissue}"
    )

    ax.text(
        1.07,
        0.05,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="0.7"),
    )

    ax.set_title("Monte Carlo Photon Trajectories (2D)", fontsize=16, fontweight="bold")
    ax.set_xlabel("X position (mm)", fontsize=13)
    ax.set_ylabel("Z depth (mm)", fontsize=13)

    ax.set_xlim(xlim)
    ax.set_ylim(tissue_depth_mm, -0.22)  # flip y-axis so depth increases downward

    ax.grid(alpha=0.25)
    ax.legend(
        bbox_to_anchor=(1.07, 1),
        loc="upper left",
        frameon=True,
        title="Photon paths",
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # For now: mock simulation output
    trajectories = generate_mock_photon_paths(
        n_photons=5000,
        max_events=30,
        step_size_mean=0.12,
        seed=7,
    )

    plot_photon_trajectories_2d(
        trajectories,
        n_photons_to_plot=10,
        max_scattering_events=10,
        total_photons=5000,
        mu_a=0.01,
        mu_s=1.0,
        g=0.90,
        n_tissue=1.4,
        tissue_depth_mm=2.0,
        xlim=(-1.0, 1.0),
        save_path="figures/photon_trajectories_2d.png",
    )
# %%
