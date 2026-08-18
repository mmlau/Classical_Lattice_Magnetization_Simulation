import cupy as cp
import numpy as np
from src.lattice import SpinLattice
from typing import Tuple

class LatticeAnalyzer:
    """
    Analyzes spin configurations with a focus on topological textures (e.g., Skyrmions).
    Computes topological charge density, total charge Q, and detects Skyrmion positions/sizes.
    Note: Some of the methods will be added with future updates
    """
    
    def __init__(self, spins: cp.ndarray):
        """
        Initializes the analyzer with the spin configuration array
        """
        self.spins = spins

    def compute_topological_charge_density(self) -> cp.ndarray:
        """
        Computes the topological charge density q(x,y) on a 2D square lattice 
        using the stored spin array and lattice neighbor methods.
        q(x,y) = M (dM/dx times dM/dy)
        """
        # Computing neighbors assuming periodic boundary conditions
        top = cp.roll(self.spins, shift=1, axis=1)
        bottom = cp.roll(self.spins, shift=-1, axis=1)
        left = cp.roll(self.spins, shift=1, axis=0)
        right = cp.roll(self.spins, shift=-1, axis=0)
        
        term_x = 0.5 * (right - left)
        term_y = 0.5 * (top - bottom)
        term_cross = cp.cross(term_x, term_y)
        density = cp.sum(self.spins * term_cross, axis=-1) / (4 * cp.pi)
        
        return density

    def compute_total_topological_charge(self) -> float:
        """
        Computes the total topological charge Q by integrating (summing) 
        the topological charge density over the entire lattice
        and scaling it with 1 / 4 pi
        """
        density = self.compute_topological_charge_density()
        Q = float(cp.sum(density))
        return Q

    def compute_position_single_skyrmion(self) -> tuple[float, float]:
        """
        Assumes one single skyrmion on the lattice and computes 
        its position based on center of mass regarding the 
        topological charge density. It checks whether a 
        skyrmion exists first.
        Attention: this fails when there are more than one skyrmion.
        """
        q_total = self.compute_total_topological_charge()

        # Avoid division by zero
        if abs(q_total) < 0.1:
            print('No skyrmion!')
            return (-1.0, -1.0)

        q = self.compute_topological_charge_density()
        Nx, Ny = q.shape
        
        # create mesh for indices
        ix_grid, iy_grid = cp.meshgrid(cp.arange(Nx), cp.arange(Ny), indexing='ij')
        
        # center of mass for topological charge density
        qx = float(cp.sum(q * ix_grid) / q_total)
        qy = float(cp.sum(q * iy_grid) / q_total)
        
        return (qx, qy)






        