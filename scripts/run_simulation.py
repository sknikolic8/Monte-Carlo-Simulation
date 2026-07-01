"""Example script: run a MC simulation and plot the results."""

import numpy as np
import matplotlib.pyplot as plt

from mcsim import TissueLayer, Simulation

# --- Tissue parameters (typical dermis values) ---
layer = TissueLayer(
    mu_a=0.1,    # 1/mm  absorption
    mu_s=10.0,   # 1/mm  scattering
    g=0.9,       #       anisotropy
    n=1.4,       #       refractive index
    depth=10.0,  # mm    slab thickness
)

sim = Simulation(layer, n_above=1.0, n_below=1.0, seed=42)
result = sim.run(n_photons=50_000)

print(f"Specular R    : {result.specular_r:.4f}")
print(f"Diffuse R     : {result.reflectance:.4f}")
print(f"Transmittance : {result.transmittance:.4f}")
print(f"Absorbed      : {result.absorbed:.4f}")
total = result.reflectance + result.transmittance + result.absorbed + result.specular_r
print(f"Energy total  : {total:.6f}  (should be ~1.0)")
assert result.check_energy_conservation(), f"Energy not conserved! total={total:.6f}"

# --- Depth-resolved absorbed energy profile ---
z_centers = 0.5 * (result.z_bins[:-1] + result.z_bins[1:])
dz = result.z_bins[1] - result.z_bins[0]
fluence = result.absorbed_profile / dz

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(z_centers, fluence, lw=1.5)
ax.set_xlabel("Depth z (mm)")
ax.set_ylabel("Absorbed energy density (mm⁻¹)")
ax.set_title(f"MC photon transport — μa={layer.mu_a}, μs={layer.mu_s}, g={layer.g}, n={layer.n}")
ax.set_xlim(0, layer.depth)
fig.tight_layout()
plt.savefig("absorbed_profile.png", dpi=150)
plt.show()
print("Plot saved to absorbed_profile.png")
