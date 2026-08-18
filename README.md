# Classical Lattice Magnetization Simulation

A lightweight, high-performance Python application for classical micromagnetic lattice simulations, accelerated via **CuPy** and the GPU.

---

## About This Project

During my PhD, I worked with my research group on developing simulation software for classical micromagnetic lattices written in **C++**. 

This repository is a private project where I use that scientific expertise to build a basic, more accessible Python version tailored for enthusiasts, researchers, and anyone interested in micromagnetism.

## What the Program Does

The software simulates the dynamic behavior of magnetic moments on a 2D square lattice by numerically integrating the **Landau-Lifshitz-Gilbert (LLG) equation** including a spin transfer torque (SST). It computes effective magnetic fields and tracks the evolution of spin states over time, utilizing GPU acceleration for efficient computation and performance.

---

## Evolving Project

Please note that I am continuously working on expanding the codebase. This includes more efficient data handling, example notebooks, and bug fixes. Since this is a private project, there is no guarantee for a specific timeline or continuous maintenance.

---

## Acknowledgments

Development of this project is carried out with the assistance of an AI collaborator to help structure code architectures and optimize workflows.

---

## Installation & Getting Started

1. Clone the repository:
```bash
git clone https://github.com/mmlau/Classical_Lattice_Magnetization_Simulation
cd Classical_Lattice_Magnetization_Simulation

2. install [uv](https://uv.dev) (if not already done) and synchronize the dependencies
```bash
# Dependencies installieren
uv sync
```