import numpy as np
import cupy as cp
from src.lattice import SpinLattice

class Hamiltonian:
    """
    Manages the physical energy terms: 
    Exchange interaction with parameter J, 
    DMI with parameter D and DMI type dmi_type bulk or if,
    External field in z direction with parameter B
    And it computes the effective field and the Landau-Lifshitz-Gilbert (LLG) dynamics with parameters:
    Gilber damping alpha,
    The precessional term gamma prime.
    """
    
    def __init__(
        self, 
        J: float = 1.0, 
        D: float = 0.0,
        B: float = 0.0,
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma_prime: float = 1.0,
        dmi_type: str = 'bulk',
        vx: float = 0.0,
        vy: float = 0.0
    ):
        if dmi_type not in ['bulk', 'if']:
            raise ValueError(f"Invalid dmi_type: '{dmi_type}'. Must be either 'bulk' or 'if'.")
        
        self.J = J
        self.D = D
        self.dmi_type = dmi_type
        self.external_field = B * cp.array([0.0, 0.0, 1.0], dtype=float)
        self.alpha = alpha
        self.beta = beta
        self.gamma_prime = gamma_prime
        self.vx = vx
        self.vy = vy

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
        # Exchange interaction
        h_eff = self.J * neighbor_sum 

        # DMI implementation based on type
        if self.dmi_type == 'bulk':
            # bulk DMI
            h_eff += 2 * self.D * cp.cross( (right - left), e_x)
            h_eff += 2 * self.D * cp.cross( (top - bottom), e_y)
        elif self.dmi_type == 'interfacial' or self.dmi_type == 'if':
            # interfacial DMI
            h_eff += 2 * self.D * cp.cross( (right - left), e_y)
            h_eff -= 2 * self.D * cp.cross( (top - bottom), e_x)

        # external magnetic field
        h_eff += self.external_field

        return h_eff

    def compute_spin_torque(self, spins: cp.ndarray, lattice: SpinLattice) -> cp.ndarray:
        """
        Computes the spin transfer torque T acting on the lattice,
        which is represented here by
        T = (alpha - beta) M x [(v . nabla) M] + (alpha beta + 1) (v . nabla) M
        """
        top, bottom, left, right = lattice.get_roll_neighbors(spins)
        v_dot_nabla_m = 0.5 * self.vx * (right - left) + 0.5 * self.vy * (top - bottom)
        spin_torque = (self.alpha - self.beta) * cp.cross( spins, v_dot_nabla_m )
        spin_torque += (self.alpha * self.beta + 1.0) * v_dot_nabla_m
        return spin_torque
    
    def llg_rhs(self, t: float, spins: cp.ndarray, lattice: SpinLattice) -> cp.ndarray:
        """
        Calculates the right-hand side (RHS) of the Landau-Lifshitz-Gilbert equation including the STT:
        dM/dt = ( -gamma_prime (M x H_eff) - alpha (M x (M x H_eff)) + T) / (1 + alpha**2),
        """
        h_eff = self.compute_effective_field(spins, lattice)
        
        # Cross products along the last axis (3D vectors)
        m_cross_h = cp.cross(spins, h_eff)
        m_cross_m_cross_h = cp.cross(spins, m_cross_h)

        # We do not always want to study externally driven skyrmions. Therefore, we
        # differentiate here between torque or no torque to be computational more efficient
        if self.vx != 0 or self.vy != 0:
            spin_torque = self.compute_spin_torque(spins, lattice)
            # Standard LLG prefactors (normalized gyromagnetic ratio terms) with STT
            dmdt = -self.gamma_prime * (m_cross_h + self.alpha * m_cross_m_cross_h + spin_torque) / (1.0 + self.alpha**2)
        else:
            # Standard LLG prefactors (normalized gyromagnetic ratio terms) without STT
            dmdt = -self.gamma_prime * (m_cross_h + self.alpha * m_cross_m_cross_h) / (1.0 + self.alpha**2)

        
        
        
        return dmdt