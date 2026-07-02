# photon_mc

Polarization-resolved Monte Carlo photon transport in a slab scattering/absorbing
medium, vectorized over the whole photon population with PyTorch tensors (runs on
GPU if available).

## Physical model

- Slab medium occupying `0 <= z <= thickness`, surrounded by index `n_outside` (air
  by default) on both sides. Both faces are real optical interfaces: a photon
  reaching either one partially reflects and partially transmits according to the
  (polarization-dependent) Fresnel equations, rather than simply escaping the
  moment it crosses the plane.
- Photons enter at the origin travelling in +z.
- Step lengths are drawn from an exponential distribution (mean = scattering mean
  free path); a step that would cross a face is clipped to the face instead, and
  Beer-Lambert absorption is applied only for the distance actually travelled.
- At the end of each step the photon scatters: the new direction is sampled from
  the configured scattering phase function (polar angle) and a uniform azimuthal
  angle, applied via an orthonormal frame (direction, polarization reference)
  carried along and updated at every scattering event. A step that hits a face
  instead undergoes a boundary interaction and skips scattering for that step.
- At a face, reflect-vs-transmit is sampled with probability equal to the actual
  polarization-dependent reflectance for the photon's current Stokes vector (not
  the unpolarized Fresnel average). Reflected photons are mirrored back into the
  medium (direction and polarization frame updated via the Fresnel reflection
  Mueller matrix, including the handedness flip reflection imparts on U/V);
  transmitted photons exit with a refracted (Snell's law) direction and the
  Fresnel transmission Mueller matrix applied to their Stokes vector. Total
  internal reflection falls out of the same complex-valued Fresnel formulas
  automatically, including the phase-retardance effect that turns linear into
  elliptical polarization on TIR.
- Each photon carries a Stokes vector `(I, Q, U, V)`, seeded from `INITIAL_STOKES`
  and evolved at each scattering event via a Mueller matrix: the configured phase
  function's own (e.g. Mie, exact for its sphere model) if it defines one,
  otherwise the generic Rayleigh approximation in `phase_functions.mueller`.
- All random sampling inverts the relevant CDF.

## Layout

```
pyproject.toml
CLAUDE.md
src/photon_mc/
    __init__.py         # package docstring + re-exports (Medium, MonteCarlo, get_phase_function)
    __main__.py          # `python -m photon_mc` entry point: runs a configured sim and plots results
    params.py             # all tunable constants (medium, phase function, polarization, transport, plotting)
    medium.py              # Medium dataclass (slab geometry + optical properties)
    simulation.py           # MonteCarlo: the batched-tensor transport loop
    plotting.py              # analysis/plot_* functions + the tabbed Tk results viewer
    phase_functions/
        __init__.py          # PHASE_FUNCTIONS registry + get_phase_function()
        henyey_greenstein.py  # HG phase function (anisotropy g)
        isotropic.py           # uniform-sphere scattering
        mie.py                  # exact Mie theory via miepython (tabulated CDF + Mueller matrix)
        mueller.py               # Rayleigh Mueller matrix, Stokes rotation, Fresnel R/T Mueller matrices
```

## Running

```
pip install -e .
python -m photon_mc      # or: photon-mc
```

Edit `src/photon_mc/params.py` to change the medium, phase function, photon count,
polarization state, etc. Results (diffuse reflectance/transmittance, angle-resolved
intensity/polarization, spatial polarization heatmaps, sample trajectories) are
printed and shown in one tabbed window.

## Adding a phase function

Drop a module in `src/photon_mc/phase_functions/` exposing a class with:

- `sample_cos_theta(xi: torch.Tensor) -> torch.Tensor` — inverts its CDF given a
  batch of `U(0, 1)` variates.
- optionally `mueller_matrix(cos_theta) -> (N, 4, 4) torch.Tensor` — used instead
  of the generic Rayleigh fallback (see `mie.py`, which derives both the sampled
  angle and the Mueller matrix from the same physical particle model).
- optionally `to(device, dtype)` if it holds precomputed tensors (see `mie.py`).

Then register it in `PHASE_FUNCTIONS` in `phase_functions/__init__.py` and select
it via `PHASE_FUNCTION` / `PHASE_FUNCTION_PARAMS` in `params.py`.
