import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data_utils.kitti_dataset import KITTIDataset
from src.data_utils.point_cloud_preprocessor import PointCloudPreprocessor
from src.models.detection_3d.pillar_feature_net import PillarFeatureNet

def test_pfn():
    # Initialize dataset and preprocessor
    dataset = KITTIDataset(
        base_path='/workspace/data/kitti/raw',
        date='2011_09_26',
        drive='0001'
    )
    preprocessor = PointCloudPreprocessor()
    
    # Load and preprocess a frame
    frame_data = dataset.get_frame_data(0)
    pillar_data = preprocessor.create_pillars(frame_data['points'])
    
    # Since the tensors are already PyTorch tensors, we just need to add batch dimension
    pillar_data = {
        'pillars': pillar_data['pillars'].unsqueeze(0),        # Add batch dimension
        'indices': pillar_data['indices'].unsqueeze(0),        # Add batch dimension
        'num_points_per_pillar': pillar_data['num_points_per_pillar'].unsqueeze(0),  # Add batch dimension
        'num_pillars': pillar_data['num_pillars']
    }
    
    # Initialize and run PFN
    pfn = PillarFeatureNet()
    pillar_features = pfn(pillar_data)
    
    # Visualize the learned features - convert to numpy for visualization
    feature_map = pillar_features[0].detach().numpy()  # (output_dim, num_pillars)
    
    plt.figure(figsize=(15, 5))
    
    # Show the first few feature channels
    plt.subplot(121)
    plt.imshow(feature_map[:10, :], aspect='auto')
    plt.title('First 10 Feature Channels')
    plt.xlabel('Pillars')
    plt.ylabel('Feature Channels')
    plt.colorbar()
    
    # Show feature statistics
    plt.subplot(122)
    plt.hist(feature_map.flatten(), bins=50)
    plt.title('Feature Value Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Output feature shape: {pillar_features.shape}")
    print(f"Feature statistics:")
    print(f"Mean: {feature_map.mean():.3f}")
    print(f"Std: {feature_map.std():.3f}")
    print(f"Min: {feature_map.min():.3f}")
    print(f"Max: {feature_map.max():.3f}")

if __name__ == "__main__":
    test_pfn()