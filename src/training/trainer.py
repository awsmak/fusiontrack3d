import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List
import time
from tqdm import tqdm

from configs.train_config import TrainingConfig

class PointPillarsTrainer:
    """Trainer class for PointPillars network."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 config: TrainingConfig,
                 train_dataset: torch.utils.data.Dataset,
                 val_dataset: torch.utils.data.Dataset = None):
        """
        Initialize trainer.
        
        Args:
            model: PointPillars model
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        self.model = model
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=self.collate_fn,
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate function for data loader.
        Combines a list of samples into a batch.
        """
        batch_dict = {
            'points': [],
            'labels': [],
            'frame_ids': []
        }
        
        for sample in batch:
            batch_dict['points'].append(torch.from_numpy(sample['points']))
            if sample['labels'] is not None:
                batch_dict['labels'].append(sample['labels'])
            batch_dict['frame_ids'].append(sample['frame_id'])
            
        # Stack points with padding
        max_points = max(points.shape[0] for points in batch_dict['points'])
        batch_size = len(batch_dict['points'])
        
        padded_points = torch.zeros(batch_size, max_points, 4)
        for i, points in enumerate(batch_dict['points']):
            padded_points[i, :points.shape[0]] = points
            
        batch_dict['points'] = padded_points
        
        return batch_dict
    
    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()
        epoch_stats = {
            'loss': 0.0,
            'cls_loss': 0.0,
            'reg_loss': 0.0,
            'dir_loss': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch_data in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Move data to device
            batch_data = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_data.items()
            }
            
            # Forward pass
            predictions = self.model(batch_data)
            
            # Compute loss
            loss_dict = self.compute_loss(predictions, batch_data)
            total_loss = (
                self.config.cls_weight * loss_dict['cls_loss'] +
                self.config.reg_weight * loss_dict['reg_loss'] +
                self.config.dir_weight * loss_dict['dir_loss']
            )
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Update statistics
            epoch_stats['loss'] += total_loss.item()
            epoch_stats['cls_loss'] += loss_dict['cls_loss'].item()
            epoch_stats['reg_loss'] += loss_dict['reg_loss'].item()
            epoch_stats['dir_loss'] += loss_dict['dir_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'cls_loss': loss_dict['cls_loss'].item(),
                'reg_loss': loss_dict['reg_loss'].item()
            })
            
        # Average statistics
        num_batches = len(self.train_loader)
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
            
        return epoch_stats
    
    def train(self, num_epochs: int):
        """Train for multiple epochs."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_stats = self.train_epoch()
            
            # Validate
            if self.val_loader:
                val_stats = self.validate()
                
                # Save best model
                if val_stats['loss'] < best_val_loss:
                    best_val_loss = val_stats['loss']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': best_val_loss
                    }, 'best_model.pth')
            
            # Update learning rate
            self.scheduler.step()