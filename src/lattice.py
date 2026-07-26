import numpy as np
import cupy as cp
from typing import Tuple

class SpinLattice:
    """
    Manages the 2D square lattice geometry, boundary conditions, 
    and the magnetic moment vectors (M_i).
    """
    
    def __init__(self, Lx: int, Ly: int):
        self.Lx = Lx
        self.Ly = Ly
        self.spins = self.initialize_random()

    def initialize_random(self) -> cp.ndarray:
        """
        Generates a normalized, random magnetization field on the 2D lattice.
        Shape: (Lx, Ly, 3)
        """
        theta = cp.random.uniform(0.0, 2.0 * cp.pi, size=(self.Lx, self.Ly))
        phi = cp.random.uniform(0.0, cp.pi, size=(self.Lx, self.Ly))
        
        mx = cp.sin(phi) * cp.cos(theta)
        my = cp.sin(phi) * cp.sin(theta)
        mz = cp.cos(phi)
        
        spins = cp.stack([mx, my, mz], axis=-1)
        return self._normalize(spins)

    def initialize_ferromagnetic(self, direction: cp.ndarray = cp.array([0.0, 0.0, 1.0])) -> cp.ndarray:
        """
        Generates a homogeneous ferromagnetic state along a specified direction.
        """
        norm_dir = direction / cp.linalg.norm(direction)
        spins = cp.zeros((self.Lx, self.Ly, 3))
        spins[..., :] = norm_dir
        return spins

    @staticmethod
    def _normalize(spins: cp.ndarray) -> cp.ndarray:
        """
        Ensures data integrity by normalizing every vector to unit length (|M_i| = 1).
        """
        norms = cp.linalg.norm(spins, axis=-1, keepdims=True)
        return spins / cp.maximum(norms, 1e-12)

    def get_roll_neighbors(self, spins: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Calculates the 4 nearest neighbors individually using periodic boundary conditions 
        via cp.roll, returning them as separate arrays (Top, Bottom, Left, Right).
        """
        neighbor_top = cp.roll(spins, shift=1, axis=0)
        neighbor_bottom = cp.roll(spins, shift=-1, axis=0)
        neighbor_left = cp.roll(spins, shift=1, axis=1)
        neighbor_right = cp.roll(spins, shift=-1, axis=1)
        
        return neighbor_top, neighbor_bottom, neighbor_left, neighbor_right