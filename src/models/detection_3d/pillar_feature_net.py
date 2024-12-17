import torch
import torch.nn as nn
from typing import Dict, Tuple

class PillarFeatureNet(nn.Module):
    """
    The Pillar Feature Network learns to create sophisticated features from raw point cloud data.
    It processes each pillar independently, transforming simple point features into rich descriptions
    that help identify objects.
    """
    def __init__(self, 
                 input_dim: int = 9,        # Our point features (x,y,z,intensity,xc,yc,zc,xp,yp)
                 hidden_dim: int = 64,      # Size of intermediate features
                 output_dim: int = 64,      # Final feature dimension
                 max_points: int = 100,     # Maximum points per pillar
                 max_pillars: int = 12000): # Maximum number of pillars
        super().__init__()
        
        # The network consists of two linear layers with batch normalization and ReLU
        # Think of this as gradually transforming our raw features into more meaningful ones
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Record the dimensions for processing
        self.max_points = max_points
        self.max_pillars = max_pillars
        self.output_dim = output_dim
        
    def forward(self, pillar_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process pillars to create enriched features.
        
        Args:
            pillar_data: Dictionary containing:
                - pillars: (B, max_pillars, max_points, features) tensor of point features
                - indices: (B, max_pillars, 2) tensor of pillar spatial indices
                - num_points_per_pillar: (B, max_pillars) tensor
                - num_pillars: Number of non-empty pillars
                
        Returns:
            Processed pillar features (B, output_dim, max_pillars)
        """
        pillars = pillar_data['pillars']  # Get the pillar features
        batch_size = pillars.shape[0]
        
        # Reshape for processing all points across all pillars together
        # This makes computation more efficient
        x = pillars.view(-1, pillars.shape[-1])  # (B * max_pillars * max_points, features)
        
        # Apply our neural network layers
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.relu(x)
        
        # Reshape back to separate pillars and points
        x = x.view(batch_size, self.max_pillars, self.max_points, self.output_dim)
        
        # Create a mask for valid points (where we have actual points, not padding)
        # This ensures we only consider real points in our calculations
        mask = torch.arange(self.max_points, device=x.device)[None, None, :] < \
               pillar_data['num_points_per_pillar'][..., None]
        mask = mask.unsqueeze(-1).expand_as(x)
        
        # Apply mask and compute mean of features for each pillar
        # This gives us one feature vector per pillar
        x = x * mask.float()
        x = x.sum(dim=2) / (pillar_data['num_points_per_pillar'][..., None] + 1e-5)
        
        # Transpose for the expected format (B, features, pillars)
        x = x.transpose(1, 2)
        
        return x