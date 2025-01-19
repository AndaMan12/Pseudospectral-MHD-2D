# 2D Pseudo-Spectral MHD Simulations

This repository contains scripts for running a **2D pseudo-spectral magnetohydrodynamics (MHD)** simulation with viscosity and magnetic diffusivity. Two main solver variants are provided:

1. **PS_runnit_vCPU.py** – A CPU-based solver using NumPy and SciPy FFT.  
2. **PS_runnit_vCuPy.py** – A GPU-based solver using CuPy, designed to leverage NVIDIA GPUs for faster computation.

Both codes evolve the **vorticity $(\omega)$** and **current density $(j)$** fields in a periodic 2D domain, subject to viscosity $\nu$ and magnetic diffusivity $\eta$.  
Additionally, there is an **analysis script** named **PS_analysis.py** to load the simulation outputs (stored in HDF5 files) and generate animations or other plots.

---

## Features
- **2D pseudo-spectral MHD** solver:
  - CPU version using NumPy & SciPy FFT.
  - GPU version using CuPy FFT.
- **Time integration** with various methods:
  - IMEX, IF, ETD (exponential integrators).
- **Volume penalization** framework (example circular mask) for modeling solid boundaries or damping.
- **HDF5** output for easy post-processing.

---
The scripts themselves attempt to install missing packages automatically via `pip install` if they are unavailable.
**Run `python3 <code_name.py> -h` to get all features and available controls.**
