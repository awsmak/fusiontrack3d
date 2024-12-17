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
from src.models.detection_3d.backbone import Backbone

def visualize_predictions(bev_features, cls_preds, box_preds):
    """Visualize network predictions."""
    plt.figure(figsize=(15, 5))
    
    # Show BEV features
    plt.subplot(131)
    mean_features = torch.mean(bev_features[0], dim=0).detach().numpy()
    plt.imshow(mean_features, cmap='viridis')
    plt.title('BEV Features')
    plt.colorbar()
    
    # Show classification predictions
    plt.subplot(132)
    cls_conf = torch.max(cls_preds[0], dim=0)[0].detach().numpy()
    plt.imshow(cls_conf, cmap='viridis')
    plt.title('Classification Confidence')
    plt.colorbar()
    
    # Show box predictions
    plt.subplot(133)
    # Visualize predicted box dimensions (width)
    box_width = box_preds[0, 3].detach().numpy()  # width channel
    plt.imshow(box_width, cmap='viridis')
    plt.title('Box Width Predictions')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def test_pipeline():
    # Initialize all components
    dataset = KITTIDataset(
        base_path='/workspace/data/kitti/raw',
        date='2011_09_26',
        drive='0001'
    )
    preprocessor = PointCloudPreprocessor()
    pfn = PillarFeatureNet()
    scatter = ScatterLayer()
    backbone = Backbone()
    
    # Load and process data
    frame_data = dataset.get_frame_data(0)
    pillar_data = preprocessor.create_pillars(frame_data['points'])
    
    # Add batch dimension
    pillar_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                   for k, v in pillar_data.items()}
    
    # Process through network
    pillar_features = pfn(pillar_data)
    bev_features = scatter(pillar_features, pillar_data['indices'])
    predictions = backbone(bev_features)
    
    # Print shapes
    print("Pipeline output shapes:")
    print(f"Pillar features: {pillar_features.shape}")
    print(f"BEV features: {bev_features.shape}")
    print(f"Classification predictions: {predictions['cls_preds'].shape}")
    print(f"Box predictions: {predictions['box_preds'].shape}")
    
    # Visualize results
    visualize_predictions(bev_features, 
                        predictions['cls_preds'],
                        predictions['box_preds'])

if __name__ == "__main__":
    test_pipeline()