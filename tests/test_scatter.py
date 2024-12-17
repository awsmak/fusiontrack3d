import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data_utils.kitti_dataset import KITTIDataset
from src.data_utils.point_cloud_preprocessor import PointCloudPreprocessor
from src.models.detection_3d.pillar_feature_net import PillarFeatureNet
from src.models.detection_3d.scatter import ScatterLayer

def visualize_bev_features(bev_features: torch.Tensor):
    """
    Visualize different aspects of the BEV feature map to understand what the network sees.
    """
    # Take the first sample from the batch
    feature_map = bev_features[0].detach().numpy()
    
    plt.figure(figsize=(15, 10))
    
    # Show mean activation across all feature channels
    plt.subplot(221)
    mean_features = np.mean(feature_map, axis=0)
    plt.imshow(mean_features, cmap='viridis')
    plt.title('Mean Feature Activation')
    plt.colorbar()
    
    # Show maximum activation across channels
    plt.subplot(222)
    max_features = np.max(feature_map, axis=0)
    plt.imshow(max_features, cmap='viridis')
    plt.title('Maximum Feature Activation')
    plt.colorbar()
    
    # Show specific feature channels
    plt.subplot(223)
    channel_idx = 0  # First channel
    plt.imshow(feature_map[channel_idx], cmap='viridis')
    plt.title(f'Channel {channel_idx} Features')
    plt.colorbar()
    
    plt.subplot(224)
    channel_idx = feature_map.shape[0] // 2  # Middle channel
    plt.imshow(feature_map[channel_idx], cmap='viridis')
    plt.title(f'Channel {channel_idx} Features')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def test_scatter():
    # Initialize all components
    dataset = KITTIDataset(
        base_path='/workspace/data/kitti/raw',
        date='2011_09_26',
        drive='0001'
    )
    preprocessor = PointCloudPreprocessor()
    pfn = PillarFeatureNet()
    scatter = ScatterLayer()
    
    # Load and process data
    frame_data = dataset.get_frame_data(0)
    pillar_data = preprocessor.create_pillars(frame_data['points'])
    
    # Add batch dimension
    pillar_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                   for k, v in pillar_data.items()}
    
    # Get pillar features
    pillar_features = pfn(pillar_data)
    
    # Create BEV feature map
    bev_features = scatter(pillar_features, pillar_data['indices'])
    
    print(f"BEV feature map shape: {bev_features.shape}")
    print("\nFeature statistics:")
    print(f"Mean: {bev_features.mean().item():.3f}")
    print(f"Std: {bev_features.std().item():.3f}")
    print(f"Min: {bev_features.min().item():.3f}")
    print(f"Max: {bev_features.max().item():.3f}")
    
    # Visualize the BEV features
    visualize_bev_features(bev_features)

if __name__ == "__main__":
    test_scatter()