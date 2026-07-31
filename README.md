# Classical Lattice Magnetization Simulation

A lightweight, high-performance Python application for classical micromagnetic lattice simulations, accelerated via **CuPy** and the GPU.

---

## About This Project

During my PhD, I worked with my research group on developing simulation software for classical micromagnetic lattices written in **C++**. 

This repository is a private project where I use that scientific expertise to build a basic, more accessible Python version tailored for enthusiasts, researchers, and anyone interested in micromagnetism.

## What the Program Does

The software simulates the dynamic behavior of magnetic moments on a 2D square lattice by numerically integrating the **Landau-Lifshitz-Gilbert (LLG) equation**. It computes effective magnetic fields (up to now only including exchange interactions and external magnetic fields) and tracks the evolution of spin states over time, utilizing GPU acceleration for efficient computation and performance.

---

## Current Status (v0.1 - Preliminary Release)

Please note that this is an **initial, preliminary version** of the software. 

* **Active Development:** I am continuously working on expanding the codebase.
* **Upcoming Features:** Future updates will introduce richer physical effects (such as spin transfer torques) and more advanced data handling routines for efficient input/output management.

---

## Acknowledgments

Development of this project is carried out with the assistance of an AI collaborator to help structure code architectures and optimize workflows.

---

## Installation & Getting Started

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/micromag-python.git](https://github.com/your-username/micromag-python.git)
   cd micromag-python

2. install [uv](https://uv.dev) (if not already done) and synchronize the dependencies
```bash
# Dependencies installieren
uv sync
```