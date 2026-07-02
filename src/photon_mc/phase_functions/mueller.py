"""Fallback Mueller-matrix scattering model for Stokes-vector polarization transport.

Used by MonteCarlo.mueller_matrix() for any phase function that doesn't
define its own (e.g. Henyey-Greenstein, isotropic): without full Mie-theory
phase matrices for an arbitrary phase function, polarization evolution is
modeled with the normalized Rayleigh (dipole) scattering matrix -- a
standard simplification used whenever exact per-particle phase matrices
aren't available. (The `Mie` phase function instead derives an exact
Mueller matrix consistent with its own sampled scattering angle -- see
phase_functions/mie.py.) The matrix is renormalized to preserve total
intensity at each event, so it only reshapes Q/U/V; photon energy
bookkeeping (Beer-Lambert attenuation, roulette) is handled separately and
is unaffected by polarization.
"""

from __future__ import annotations

import torch


def rayleigh_mueller_matrix(cos_theta: torch.Tensor) -> torch.Tensor:
    """Batched Rayleigh (dipole) scattering Mueller matrices, shape (N, 4, 4).

    Defined in the scattering-plane basis: apply to a Stokes vector that has
    already been rotated into the scattering plane (see `rotate_stokes`).
    """
    c = cos_theta
    c2 = c * c
    zero = torch.zeros_like(c)

    row0 = torch.stack([0.5 * (c2 + 1.0), 0.5 * (c2 - 1.0), zero, zero], dim=-1)
    row1 = torch.stack([0.5 * (c2 - 1.0), 0.5 * (c2 + 1.0), zero, zero], dim=-1)
    row2 = torch.stack([zero, zero, c, zero], dim=-1)
    row3 = torch.stack([zero, zero, zero, c], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)  # (N, 4, 4)


def rotate_stokes(stokes: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Rotate batched Stokes vectors (N, 4) into the scattering plane defined
    by azimuth `phi` (radians) about the propagation direction."""
    cos2p, sin2p = torch.cos(2.0 * phi), torch.sin(2.0 * phi)
    i, q, u, v = stokes.unbind(-1)
    q_rot = cos2p * q + sin2p * u
    u_rot = -sin2p * q + cos2p * u
    return torch.stack([i, q_rot, u_rot, v], dim=-1)


def fresnel_mueller_matrices(cos_i: torch.Tensor, n1: float, n2: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched Fresnel reflection and transmission Mueller matrices (N, 4, 4)
    for a flat interface between real refractive indices n1 (incident side)
    and n2 (transmitted side), given cos(angle of incidence) `cos_i`.

    Defined in the (p, s) plane-of-incidence basis: apply to a Stokes vector
    that has already been rotated so Q is aligned with p (see `rotate_stokes`
    and its use in MonteCarlo.run()'s boundary handling). Uses complex-valued
    amplitude coefficients throughout, which -- with no special-casing --
    also correctly reproduces total internal reflection: energy reflectance
    becomes exactly 1, and the p/s phase difference correctly rotates U into
    V (the same Fresnel-rhomb effect that turns linear into elliptical
    polarization on TIR), instead of naively clamping to an unpolarized R=1.

    Returns (R, T, cos_t) where cos_t is the (real) cosine of the refraction
    angle -- meaningless where TIR occurs, but unused there since reflectance
    is exactly 1 in that regime.
    """
    complex_dtype = torch.complex128 if cos_i.dtype == torch.float64 else torch.complex64
    cos_i_r = cos_i.clamp(-1.0, 1.0)
    sin_i2 = (1.0 - cos_i_r * cos_i_r).clamp_min(0.0)
    sin_t2 = (n1 / n2) ** 2 * sin_i2
    cos_i_c = cos_i_r.to(complex_dtype)
    cos_t = torch.sqrt((1.0 - sin_t2).to(complex_dtype))  # principal branch: real (propagating) or +i*|.| (evanescent/TIR)

    rs = (n1 * cos_i_c - n2 * cos_t) / (n1 * cos_i_c + n2 * cos_t)
    rp = (n2 * cos_i_c - n1 * cos_t) / (n2 * cos_i_c + n1 * cos_t)
    ts = (2.0 * n1 * cos_i_c) / (n1 * cos_i_c + n2 * cos_t)
    tp = (2.0 * n1 * cos_i_c) / (n2 * cos_i_c + n1 * cos_t)

    Rs, Rp = rs.abs() ** 2, rp.abs() ** 2
    power_ratio = (n2 * cos_t.real) / (n1 * cos_i_r.clamp_min(1e-12))  # flux, not amplitude, transmittance factor
    Ts, Tp = power_ratio * ts.abs() ** 2, power_ratio * tp.abs() ** 2

    delta_r = torch.angle(rp) - torch.angle(rs)
    delta_t = torch.angle(tp) - torch.angle(ts)

    def _matrix(par: torch.Tensor, perp: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        cross = torch.sqrt((par * perp).clamp_min(0.0))
        cos_d, sin_d = torch.cos(delta), torch.sin(delta)
        zero = torch.zeros_like(par)
        row0 = torch.stack([0.5 * (par + perp), 0.5 * (par - perp), zero, zero], dim=-1)
        row1 = torch.stack([0.5 * (par - perp), 0.5 * (par + perp), zero, zero], dim=-1)
        row2 = torch.stack([zero, zero, cross * cos_d, cross * sin_d], dim=-1)
        row3 = torch.stack([zero, zero, -cross * sin_d, cross * cos_d], dim=-1)
        return torch.stack([row0, row1, row2, row3], dim=-2)

    R = _matrix(Rp, Rs, delta_r)
    T = _matrix(Tp, Ts, delta_t)
    return R, T, cos_t.real
