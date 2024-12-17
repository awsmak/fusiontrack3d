import torch
import torch.nn as nn
from typing import Dict, Tuple

class ScatterLayer(nn.Module):
    """
    The Scatter layer takes processed pillar features and creates a dense bird's eye view (BEV) 
    feature map. Imagine taking all the information we learned about each pillar and placing it 
    back onto a top-down view of the scene, creating something like a detailed aerial map.
    """
    def __init__(self, nx: int = 625, ny: int = 625):
        """
        Initialize the scatter layer with grid dimensions.
        
        Args:
            nx: Number of cells along x-axis in the BEV grid
            ny: Number of cells along y-axis in the BEV grid
        """
        super().__init__()
        self.nx = nx
        self.ny = ny
    
    def forward(self, 
                pillar_features: torch.Tensor, 
                coords: torch.Tensor) -> torch.Tensor:
        """
        Create a dense BEV feature map by scattering pillar features to their corresponding
        positions in the grid.
        
        Args:
            pillar_features: Tensor of shape (batch_size, num_features, num_pillars)
                           containing the processed features for each pillar
            coords: Tensor of shape (batch_size, num_pillars, 2) containing the x,y
                   indices for each pillar in the BEV grid
        
        Returns:
            BEV feature map of shape (batch_size, num_features, ny, nx)
        """
        batch_size = pillar_features.shape[0]
        num_features = pillar_features.shape[1]
        
        # Create empty canvas for our BEV feature map
        # Think of this as a blank sheet where we'll draw our feature map
        bev_map = torch.zeros(
            (batch_size, num_features, self.ny, self.nx),
            dtype=pillar_features.dtype,
            device=pillar_features.device
        )
        
        # For each sample in the batch
        for batch_idx in range(batch_size):
            # The indexing operation below places each pillar's features at its corresponding
            # position in the BEV grid. It's like placing each piece of information exactly
            # where it belongs on our map.
            bev_map[batch_idx, :, coords[batch_idx, :, 1], coords[batch_idx, :, 0]] = \
                pillar_features[batch_idx]
        
        return bev_map