import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """
    Handles the graphical representation of the 2D magnetic moment field,
    including vector field plots and potential future animations.
    """
    
    @staticmethod
    def plot_vector_field(spins: np.ndarray, title: str = "Magnetization Field") -> None:
        """
        Plots the in-plane magnetization components (mx, my) as a quiver (vector) plot 
        and the out-of-plane component (mz) as a colormap background.
        """
        mx = spins[..., 0].T
        my = spins[..., 1].T
        mz = spins[..., 2].T
        
        Ly, Lx = spins.shape[:2]
        x = np.arange(Lx)
        y = np.arange(Ly)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(7, 6))
        plt.title(title)
        
        # Background colormap for Mz component
        im = plt.imshow(mz, origin='lower', cmap='coolwarm', vmin=-1, vmax=-1 if np.all(mz==1) else 1)
        plt.colorbar(im, label='Mz Component')
        
        # Quiver plot for In-Plane components (Mx, My)
        plt.quiver(X, Y, mx, my, color='black', scale=max(Lx, Ly) / 2.0, width=0.003)
        
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.tight_layout()
        plt.show()

    def plot_scalar_field(field: np.ndarray, title: str = "Topological Charge Density", cmap: str = "coolwarm"):
        """
        Plots a 2D scalar field (e.g., topological charge density) using Matplotlib.
        
        Parameters:
        - field: Numpy array of shape (Lx, Ly) representing the scalar field.
        - title: Title of the plot.
        - cmap: Colormap for the heatmap (e.g., 'coolwarm', 'viridis', 'seismic').
        """

        plt.figure(figsize=(6, 5))
        im = plt.imshow(field.T, origin='lower', cmap=cmap)
        
        plt.colorbar(im, label="Density")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.show()