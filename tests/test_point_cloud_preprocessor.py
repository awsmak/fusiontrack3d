import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.data_utils.kitti_dataset import KITTIDataset
from src.data_utils.point_cloud_preprocessor import PointCloudPreprocessor

def visualize_pillars(points, preprocessed_data, x_range, y_range):
    """Visualize original points and pillar grid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original points
    ax1.scatter(points[:, 0], points[:, 1], c=points[:, 2], 
                cmap='viridis', s=1, alpha=0.5)
    ax1.set_title('Original Point Cloud (Top View)')
    ax1.set_xlim(x_range[0], x_range[1])
    ax1.set_ylim(y_range[0], y_range[1])
    ax1.set_aspect('equal')
    
    # Plot pillar centers
    indices = preprocessed_data['indices']
    x_step, y_step = x_range[2], y_range[2]
    pillar_x = indices[:preprocessed_data['num_pillars'], 0] * x_step + x_range[0]
    pillar_y = indices[:preprocessed_data['num_pillars'], 1] * y_step + y_range[0]
    
    points_per_pillar = preprocessed_data['num_points_per_pillar'][:preprocessed_data['num_pillars']]
    
    scatter = ax2.scatter(pillar_x, pillar_y, 
                     c=points_per_pillar, 
                     cmap='plasma',  # Try different colormap
                     s=50, 
                     alpha=0.8)
    ax2.set_title('Pillar Grid')
    ax2.set_xlim(x_range[0], x_range[1])
    ax2.set_ylim(y_range[0], y_range[1])
    ax2.set_aspect('equal')
    
    plt.colorbar(scatter, ax=ax2, label='Points per Pillar')
    plt.tight_layout()
    plt.show()

def test_preprocessor():
    # Initialize dataset
    dataset = KITTIDataset(
        base_path='/workspace/data/kitti/raw',
        date='2011_09_26',
        drive='0001'
    )
    
    # Load a frame
    frame_data = dataset.get_frame_data(0)
    points = frame_data['points']
    
    # Initialize preprocessor
    preprocessor = PointCloudPreprocessor()
    
    # Process point cloud
    preprocessed_data = preprocessor.create_pillars(points)
    
    # Visualize results
    print(f"Total points: {len(points)}")
    print(f"Number of pillars: {preprocessed_data['num_pillars']}")
    print(f"Maximum points in a pillar: {preprocessed_data['num_points_per_pillar'].max()}")
    
    visualize_pillars(points, preprocessed_data,
                     preprocessor.x_range, preprocessor.y_range)

if __name__ == "__main__":
    test_preprocessor()