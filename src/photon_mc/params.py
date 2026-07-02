"""Default simulation parameters.

Edit these to configure a run of `python -m photon_mc`; every value here is
also a valid keyword default for the corresponding `Medium`/`MonteCarlo`
argument, so scripts can override individual parameters without touching
this file.
"""

from __future__ import annotations

import torch

# ----- medium / geometry --------------------------------------------------

MU_A = 0.1                  # absorption coefficient [1/mm]
MU_S = 10.0                 # scattering coefficient [1/mm] (all other length quantities in the
                             # simulation -- positions, step lengths, plot axes -- are in mm too)
N_MEDIUM = 1.4               # refractive index of the medium (dimensionless)
N_OUTSIDE = 1.0              # refractive index of the surrounding material outside both faces
                             # (air = 1.0); together with N_MEDIUM sets the Fresnel reflectance/
                             # refraction at z=0 and z=SLAB_THICKNESS -- same index assumed on
                             # both faces
SLAB_THICKNESS = 5.0         # [mm] the medium occupies 0 <= z <= SLAB_THICKNESS; the bottom face
                             # is a real interface too (partial reflection/transmission), not a
                             # hard cutoff

# ----- phase function -------------------------------------------------------

#PHASE_FUNCTION = "henyey_greenstein"     # see phase_functions.PHASE_FUNCTIONS for options
#PHASE_FUNCTION_PARAMS = {"g": 0.9}       # kwargs passed to the phase function's constructor
# "mie" instead uses exact Mie theory (via miepython) for both the sampled
# scattering angle and its Mueller matrix, from the sphere's actual physical
# size rather than an abstract size parameter, e.g. a ~1um-diameter,
# water-like (n=1.33) particle illuminated by a HeNe laser (632.8nm in vacuum):
PHASE_FUNCTION = "mie"
PHASE_FUNCTION_PARAMS = {
    "m": 1.33 - 0.0j,        # refractive index of the particle relative to the medium (dimensionless)
    "radius": 0.5,           # particle radius [um]
    "wavelength": 0.633,     # wavelength of light in the medium [um]
}

# ----- polarization ----------------------------------------------------------

INITIAL_STOKES = (1.0, 1.0, 0.0, 0.0)    # (I, Q, U, V) at launch; (1,0,0,0)=unpolarized,
                                          # (1,1,0,0)/(1,-1,0,0)=horiz/vert linear,
                                          # (1,0,1,0)=+45deg linear, (1,0,0,1)=right circular

# If True, the scattering angle (theta, phi) is drawn from the TRUE
# polarization-dependent scattered intensity for the photon's current Stokes
# vector, instead of sample_cos_theta()'s unpolarized S11-only phase function
# + a phi uniform in [0, 2pi). Uses whichever Mueller matrix
# MonteCarlo.mueller_matrix() would use anyway (Mie's exact one, or the
# Rayleigh fallback for HG/isotropic). See MonteCarlo._sample_scattering_angle_polarized.
USE_POLARIZED_SAMPLING = True

# HEALPix grid used by _sample_scattering_angle_polarized to discretize the
# polarization-dependent scattered intensity: a sharply forward-peaked phase
# function (e.g. large-particle Mie) needs a finer grid to resolve the
# forward lobe. npix = 12 * HEALPIX_NSIDE**2.
HEALPIX_NSIDE = 16
# Photons are processed HEALPIX_CHUNK_SIZE at a time when building the
# per-photon (chunk, npix) weight/CDF matrix, so peak memory stays bounded
# regardless of N_PHOTONS or HEALPIX_NSIDE.
HEALPIX_CHUNK_SIZE = 100_000

# ----- transport / roulette --------------------------------------------------

WEIGHT_THRESHOLD = 1e-4     # roulette trigger
SURVIVAL_CHANCE = 0.01      # roulette survival probability

N_PHOTONS = 2_000_000       # batching makes large populations cheap on a GPU
MAX_STEPS = 10000           # max steps per photon before giving up

# ----- plotting ---------------------------------------------------------------

N_ANGLE_BINS = 60           # number of bins for the backscatter-angle plot
N_SPATIAL_BINS = 40         # per-axis resolution of the (x, y) polarization heatmap
AZIMUTHAL_PLOT_THETA_DEG = 90.0   # polar angle at which the azimuthal scattering-probability plot is evaluated

N_TRAJECTORY_PHOTONS = 2000   # small dedicated batch run with full position history recorded
N_TRAJECTORIES_TO_PLOT = 12   # how many backscattered trajectories from that batch to draw

# ----- runtime -----------------------------------------------------------------

RNG_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32       # most GPUs run float32 far faster than float64
