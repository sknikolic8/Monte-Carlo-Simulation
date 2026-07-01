"""Command-line entry point for mcsim."""

from __future__ import annotations

import argparse

from .tissue import TissueLayer
from .simulation import Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo photon transport in tissue")
    parser.add_argument("--n-photons", type=int, default=10_000, metavar="N")
    parser.add_argument("--mu-a", type=float, default=0.1, help="Absorption coeff (1/mm)")
    parser.add_argument("--mu-s", type=float, default=10.0, help="Scattering coeff (1/mm)")
    parser.add_argument("--g", type=float, default=0.9, help="Anisotropy factor")
    parser.add_argument("--n", type=float, default=1.4, help="Refractive index of tissue")
    parser.add_argument("--depth", type=float, default=10.0, help="Slab thickness (mm)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="Show absorbed-energy depth profile")
    args = parser.parse_args()

    layer = TissueLayer(mu_a=args.mu_a, mu_s=args.mu_s, g=args.g, n=args.n, depth=args.depth)
    sim = Simulation(layer, seed=args.seed)
    result = sim.run(args.n_photons)

    print(f"Photons          : {result.n_photons:,}")
    print(f"Specular R       : {result.specular_r:.4f}")
    print(f"Diffuse R        : {result.reflectance:.4f}")
    print(f"Transmittance    : {result.transmittance:.4f}")
    print(f"Absorbed         : {result.absorbed:.4f}")
    total = result.reflectance + result.transmittance + result.absorbed + result.specular_r
    print(f"Energy check     : {total:.6f}  (should be 1.0)")

    if args.plot:
        _plot(result)


def _plot(result) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    z_centers = 0.5 * (result.z_bins[:-1] + result.z_bins[1:])
    dz = result.z_bins[1] - result.z_bins[0]
    fluence = result.absorbed_profile / dz

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z_centers, fluence)
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Absorbed energy density (mm⁻¹)")
    ax.set_title("Depth-resolved absorbed energy")
    fig.tight_layout()
    plt.show()
