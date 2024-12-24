import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class PointPillarsLoss(nn.Module):
    """
    Combined loss function for 3D object detection with PointPillars.
    Includes classification loss, box regression loss, and direction classification loss.
    """
    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss.
        
        Args:
            predictions: Dictionary containing network predictions
                - cls_preds: Classification predictions
                - box_preds: Box regression predictions
            targets: Dictionary containing target values
                - cls_targets: Classification targets
                - box_targets: Box regression targets
                - cls_weights: Classification weights
                - box_weights: Box regression weights
        """
        cls_preds = predictions['cls_preds']
        box_preds = predictions['box_preds']
        cls_targets = targets['cls_targets']
        box_targets = targets['box_targets']
        
        # Classification loss (Focal Loss)
        cls_loss = self.focal_loss(
            cls_preds, 
            cls_targets, 
            weights=targets.get('cls_weights', None)
        )
        
        # Box regression loss (Smooth L1)
        box_loss = self.smooth_l1_loss(
            box_preds,
            box_targets,
            weights=targets.get('box_weights', None)
        )
        
        return {
            'cls_loss': cls_loss,
            'reg_loss': box_loss,
            'total_loss': cls_loss + box_loss
        }
    
    def focal_loss(self, 
                   predictions: torch.Tensor,
                   targets: torch.Tensor,
                   weights: torch.Tensor = None,
                   gamma: float = 2.0,
                   alpha: float = 0.25) -> torch.Tensor:
        """
        Compute Focal Loss for better handling of class imbalance.
        """
        predictions = torch.sigmoid(predictions)
        
        # Compute focal weight
        pt = predictions * targets + (1 - predictions) * (1 - targets)
        focal_weight = (1 - pt) ** gamma
        
        # Compute alpha weight
        alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
        
        # Compute binary cross entropy
        bce = -torch.log(pt + 1e-6)  # Add small epsilon for numerical stability
        
        loss = focal_weight * alpha_weight * bce
        
        # Apply additional weights if provided
        if weights is not None:
            loss = loss * weights
            
        return loss.mean()
    
    def smooth_l1_loss(self,
                      predictions: torch.Tensor,
                      targets: torch.Tensor,
                      weights: torch.Tensor = None,
                      beta: float = 0.1) -> torch.Tensor:
        """
        Compute Smooth L1 Loss for box regression.
        """
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        
        # Compute smooth L1
        loss = torch.where(
            abs_diff < beta,
            0.5 * diff * diff / beta,
            abs_diff - 0.5 * beta
        )
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
            
        return loss.mean()