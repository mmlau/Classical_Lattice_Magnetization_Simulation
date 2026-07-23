import numpy as np
import os

class IOHandler:
    """
    Handles data persistence, such as saving simulation checkpoints 
    and ensuring logical consistency of output data.
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_checkpoint(self, step: int, time: float, spins: np.ndarray) -> str:
        """
        Saves the current system state (step, time, and spin configuration) 
        as a compressed NumPy archive (.npz).
        """
        filename = os.path.join(self.output_dir, f"checkpoint_step_{step:05d}.npz")
        np.savez_compressed(filename, step=step, time=time, spins=spins)
        print(f"Checkpoint saved: {filename}")
        return filename

    def load_checkpoint(self, filename: str) -> dict:
        """
        Loads a previously saved simulation checkpoint.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint file not found: {filename}")
            
        data = np.load(filename)
        return {
            "step": int(data["step"]),
            "time": float(data["time"]),
            "spins": data["spins"]
        }