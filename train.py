import torch
from torch.utils.data import DataLoader
import os
from typing import Dict
from tqdm import tqdm

from src.data_utils.kitti_object_dataset import KITTIObjectDataset
from src.data_utils.augmentation import PointCloudAugmentor
from src.data_utils.point_cloud_preprocessor import PointCloudPreprocessor
from src.models.detection_3d.pillar_feature_net import PillarFeatureNet
from src.models.detection_3d.scatter import ScatterLayer
from src.models.detection_3d.backbone import Backbone
from src.training.losses import PointPillarsLoss
from configs.train_config import TrainingConfig

def train_model(config: TrainingConfig):
    """
    Main training function.
    """
    # Initialize datasets
    train_dataset = KITTIObjectDataset(
        base_path=config.data_path,
        date='2011_09_26',
        drive='0001'
    )
    
    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model components
    preprocessor = PointCloudPreprocessor()
    pillar_net = PillarFeatureNet()
    scatter = ScatterLayer()
    backbone = Backbone()
    
    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pillar_net = pillar_net.to(device)
    backbone = backbone.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam([
        {'params': pillar_net.parameters()},
        {'params': backbone.parameters()}
    ], lr=config.learning_rate)
    
    # Initialize loss function
    criterion = PointPillarsLoss()
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Training phase
        pillar_net.train()
        backbone.train()
        
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Preprocess point clouds
            processed_data = process_batch(batch_data, preprocessor, device)
            
            # Forward pass
            pillar_features = pillar_net(processed_data)
            bev_features = scatter(pillar_features, processed_data['indices'])
            predictions = backbone(bev_features)
            
            # Compute loss
            loss_dict = criterion(predictions, processed_data)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
            
        # Print epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                epoch,
                pillar_net,
                backbone,
                optimizer,
                avg_loss,
                config
            )

def process_batch(batch_data: Dict, 
                 preprocessor: PointCloudPreprocessor,
                 device: torch.device) -> Dict:
    """Process a batch of point clouds."""
    processed_batch = {
        'points': [],
        'indices': [],
        'num_points': []
    }
    
    for points in batch_data['points']:
        # Apply preprocessing
        pillar_data = preprocessor.create_pillars(points.numpy())
        
        processed_batch['points'].append(pillar_data['pillars'])
        processed_batch['indices'].append(pillar_data['indices'])
        processed_batch['num_points'].append(pillar_data['num_points_per_pillar'])
    
    # Stack batch data
    processed_batch = {
        'points': torch.stack(processed_batch['points']).to(device),
        'indices': torch.stack(processed_batch['indices']).to(device),
        'num_points': torch.stack(processed_batch['num_points']).to(device)
    }
    
    return processed_batch

def save_checkpoint(epoch: int,
                   pillar_net: torch.nn.Module,
                   backbone: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   loss: float,
                   config: TrainingConfig):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(config.data_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f'checkpoint_epoch_{epoch+1}.pth'
    )
    
    torch.save({
        'epoch': epoch,
        'pillar_net_state_dict': pillar_net.state_dict(),
        'backbone_state_dict': backbone.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    config = TrainingConfig()
    train_model(config)