import numpy as np
import cupy as cp
from src.lattice import SpinLattice

class Hamiltonian:
    """
    Manages the physical energy terms: exchange interaction, external field, DMI
    and computes the effective field and the Landau-Lifshitz-Gilbert (LLG) dynamics.
    """
    
    def __init__(
        self, 
        J: float = 1.0, 
        D: float = 0.18,
        external_field: cp.ndarray = cp.array([0.0, 0.0, 0.1], dtype=float), 
        alpha: float = 0.2
    ):
        self.J = J  # Exchange constant
        self.D = D # DMI constant
        self.external_field = external_field  # External potential / magnetic field
        self.alpha = alpha  # Gilbert damping parameter

    def compute_effective_field(self, spins: cp.ndarray, lattice: SpinLattice) -> cp.ndarray:
        """
        Computes the effective field H_eff = - dE / dM at each lattice site,
        incorporating nearest-neighbor exchange, DMI, and the external field.
        """

        # unity vectors
        e_x = cp.array([1.0, 0.0, 0.0])
        e_y = cp.array([0.0, 1.0, 0.0])
        top, bottom, left, right = lattice.get_roll_neighbors(spins)
        neighbor_sum = top + bottom + left + right
        
        # H_eff = J * sum(M_neighbors) + DMI + B_ext
        h_eff = self.J * neighbor_sum 

        # Using bulk DMI for now
        h_eff += 2 * self.D * cp.cross( (right - left), e_x)
        h_eff += 2 * self.D * cp.cross( (top - bottom), e_y)
        h_eff += self.external_field


        return h_eff

    def llg_rhs(self, t: float, spins: cp.ndarray, lattice: SpinLattice) -> cp.ndarray:
        """
        Calculates the right-hand side (RHS) of the Landau-Lifshitz-Gilbert equation:
        dM/dt = -gamma_prime (M x H_eff) - alpha * gamma_prime (M x (M x H_eff))
        """
        h_eff = self.compute_effective_field(spins, lattice)
        
        # Cross products along the last axis (3D vectors)
        m_cross_h = cp.cross(spins, h_eff)
        m_cross_m_cross_h = cp.cross(spins, m_cross_h)
        
        # Standard LLG prefactors (normalized gyromagnetic ratio terms)
        gamma_prime = 1.0 / (1.0 + self.alpha**2)
        dmdt = -gamma_prime * (m_cross_h + self.alpha * m_cross_m_cross_h)
        
        return dmdt