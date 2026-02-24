# cse5280-multiple-particles-pkelly2022
Multi-Particle animation 

# CSE5280 – Multi-Particle Simulation with Social Forces
Patrick Kelly
NetID: pkelly2022

## How to Run

python multiple_particles.py

Toggle:
use_anisotropic = False  -> Isotropic model
use_anisotropic = True   -> Anisotropic model

---

## Models Implemented

1. Isotropic quadratic personal-space repulsion
2. Anisotropic velocity-dependent social force

Both models use the same base distance penalty. The anisotropic model applies a directional weighting based on velocity.

---

## Comparison of Isotropic vs Anisotropic

Isotropic forces are symmetric. The repulsion depends only on distance, so particles react equally in all directions. In narrow corridors or bottlenecks, this symmetry can cause oscillations or temporary deadlocks because particles push equally against each other.

The anisotropic model breaks this symmetry by prioritizing interactions in the direction of motion. Particles are more sensitive to agents in front of them than behind them. This reduces head-on oscillation behavior and produces smoother flow through the bottleneck.

In the corridor experiment, the isotropic model showed more symmetric clustering, while the anisotropic model produced more directional flow and slightly reduced congestion.

Trade-off: the anisotropic model is more realistic but depends on velocity, making behavior more sensitive to parameter tuning (beta).
