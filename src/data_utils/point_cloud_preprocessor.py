import numpy as np
from typing import Dict, Tuple, List, Optional
import torch

class PointCloudPreprocessor:
    """
    Preprocesses raw LiDAR point clouds for PointPillars network.
    Converts irregular point cloud data into a regular grid of pillars.
    """
    def __init__(self, 
                 x_range: Tuple[float, float, float] = (-50, 50, 0.16),  # (min, max, step)
                 y_range: Tuple[float, float, float] = (-50, 50, 0.16),
                 z_range: Tuple[float, float, float] = (-3, 3, 0.2),
                 max_points_per_pillar: int = 100,
                 max_pillars: int = 12000):
        """
        Initialize preprocessor with grid parameters.
        
        Args:
            x_range: Range and resolution in x dimension (min, max, step)
            y_range: Range and resolution in y dimension
            z_range: Range and resolution in z dimension
            max_points_per_pillar: Maximum number of points per pillar
            max_pillars: Maximum number of non-empty pillars
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars
        
        # Calculate grid size
        self.nx = int((x_range[1] - x_range[0]) / x_range[2])
        self.ny = int((y_range[1] - y_range[0]) / y_range[2])
        self.nz = int((z_range[1] - z_range[0]) / z_range[2])
        
        print(f"Grid size: {self.nx} x {self.ny} x {self.nz}")
    
    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """Filter points based on range settings."""
        mask = (
            (points[:, 0] >= self.x_range[0]) & (points[:, 0] <= self.x_range[1]) &
            (points[:, 1] >= self.y_range[0]) & (points[:, 1] <= self.y_range[1]) &
            (points[:, 2] >= self.z_range[0]) & (points[:, 2] <= self.z_range[1])
        )
        return points[mask]
    
    def _create_pillar_indices(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert points to pillar indices."""
        x_indices = ((points[:, 0] - self.x_range[0]) / self.x_range[2]).astype(np.int32)
        y_indices = ((points[:, 1] - self.y_range[0]) / self.y_range[2]).astype(np.int32)
        
        # Combine indices into unique pillar IDs
        pillar_indices = y_indices * self.nx + x_indices
        
        return pillar_indices, np.column_stack([x_indices, y_indices])
    
    def create_pillars(self, points: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Convert point cloud to pillar format.
        
        Args:
            points: (N, 4) array of points (x, y, z, intensity)
            
        Returns:
            Dictionary containing:
            - pillars: (max_pillars, max_points_per_pillar, 9) tensor of pillar features
            - indices: (max_pillars, 2) tensor of pillar spatial indices
            - num_points_per_pillar: (max_pillars,) tensor
        """
        # Filter points within range
        points = self._filter_points(points)
        
        # Get pillar indices
        pillar_indices, spatial_indices = self._create_pillar_indices(points)
        
        # Initialize output tensors
        pillars = torch.zeros((self.max_pillars, self.max_points_per_pillar, 9), 
                            dtype=torch.float32)
        indices = torch.zeros((self.max_pillars, 2), dtype=torch.int32)
        num_points_per_pillar = torch.zeros(self.max_pillars, dtype=torch.int32)
        
        # Process each unique pillar
        unique_indices, counts = np.unique(pillar_indices, return_counts=True)
        num_pillars = min(len(unique_indices), self.max_pillars)
        
        for i in range(num_pillars):
            pillar_id = unique_indices[i]
            points_mask = pillar_indices == pillar_id
            pillar_points = points[points_mask]
            
            # Limit number of points per pillar
            num_points = min(len(pillar_points), self.max_points_per_pillar)
            pillar_points = pillar_points[:num_points]
            
            # Calculate pillar features
            # [x, y, z, intensity, x_c, y_c, z_c, x_p, y_p]
            x_c = pillar_points[:, 0].mean()
            y_c = pillar_points[:, 1].mean()
            z_c = pillar_points[:, 2].mean()
            
            x_p = float(spatial_indices[points_mask][0][0] * self.x_range[2] + self.x_range[0])
            y_p = float(spatial_indices[points_mask][0][1] * self.y_range[2] + self.y_range[0])
            
            features = np.column_stack([
                pillar_points,  # original features [x, y, z, intensity]
                np.full((num_points, 3), [x_c, y_c, z_c]),  # center coordinates
                np.full((num_points, 2), [x_p, y_p])  # pillar coordinates
            ])
            
            pillars[i, :num_points] = torch.from_numpy(features)
            indices[i] = torch.from_numpy(spatial_indices[points_mask][0])
            num_points_per_pillar[i] = num_points
        
        return {
            'pillars': pillars,
            'indices': indices,
            'num_points_per_pillar': num_points_per_pillar,
            'num_pillars': num_pillars
        }