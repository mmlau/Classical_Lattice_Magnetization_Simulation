import numpy as np
from src.lattice import SpinLattice
from src.hamiltonian import Hamiltonian

class RungeKutta4Solver:
    """
    Numerical time integrator using a 4th-order Runge-Kutta (RK4) scheme 
    tailored for vector fields on the lattice.
    """
    
    def __init__(self, lattice: SpinLattice, hamiltonian: Hamiltonian):
        self.lattice = lattice
        self.hamiltonian = hamiltonian

    def step(self, t: float, spins: np.ndarray, dt: float) -> np.ndarray:
        """
        Performs a single RK4 time step for the entire magnetization field.
        """
        # k1 stage
        k1 = self.hamiltonian.llg_rhs(t, spins, self.lattice)
        
        # k2 stage
        spins_k2 = self.lattice._normalize(spins + 0.5 * dt * k1)
        k2 = self.hamiltonian.llg_rhs(t + 0.5 * dt, spins_k2, self.lattice)
        
        # k3 stage
        spins_k3 = self.lattice._normalize(spins + 0.5 * dt * k2)
        k3 = self.hamiltonian.llg_rhs(t + 0.5 * dt, spins_k3, self.lattice)
        
        # k4 stage
        spins_k4 = self.lattice._normalize(spins + dt * k3)
        k4 = self.hamiltonian.llg_rhs(t + dt, spins_k4, self.lattice)
        
        # Combine stages for the final update
        new_spins = spins + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # Enforce strict data integrity: normalize all vectors back to unit length
        return self.lattice._normalize(new_spins)