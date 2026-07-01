# Monte Carlo Tissue Simulation


Energy conserves to 6 decimal places. Here's the final layout:

  Monte-Carlo-Simulation/
  ├── pyproject.toml               # hatchling build, mcsim-run console entry
  ├── scripts/
  │   └── run_simulation.py        # example: run 50k photons, plot + save PNG
  └── src/
      └── mcsim/
          ├── __init__.py
          ├── tissue.py            # TissueLayer dataclass (μa, μs, g, n, depth)
          ├── simulation.py        # MCML photon loop + Fresnel/HG physics
          └── cli.py               # argparse CLI (mcsim-run command)