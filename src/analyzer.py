import cupy as cp
import numpy as np
from src.lattice import SpinLattice

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
        density = cp.sum(self.spins * term_cross, axis=-1)
        
        return density

    def compute_total_topological_charge(self) -> float:
        """
        Computes the total topological charge Q by integrating (summing) 
        the topological charge density over the entire lattice
        and scaling it with 1 / 4 pi
        """
        density = self.compute_topological_charge_density()
        Q = float(cp.sum(density)) / (4 * cp.pi)
        return Q