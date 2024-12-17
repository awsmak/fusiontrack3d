import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class Backbone(nn.Module):
    """
    Backbone network for 3D object detection.
    Uses a series of convolutional blocks to process the BEV feature map
    and produce detection predictions.
    """
    def __init__(self, 
                 input_channels: int = 64,     # Number of input feature channels
                 layer_channels: List[int] = [64, 128, 256],  # Channels in each layer
                 layer_strides: List[int] = [2, 2, 2]):       # Strides for each layer
        super().__init__()
        
        # Validation
        assert len(layer_channels) == len(layer_strides), \
            "Number of channels must match number of strides"
        
        self.blocks = nn.ModuleList()
        in_channels = input_channels
        
        # Create convolutional blocks
        for out_channels, stride in zip(layer_channels, layer_strides):
            self.blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride
                )
            )
            in_channels = out_channels
            
        # Detection head
        self.detect_head = DetectionHead(
            in_channels=layer_channels[-1],
            num_classes=3,  # Car, Pedestrian, Cyclist
            num_anchors=2   # Two anchor boxes per position
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process BEV features through the backbone network.
        
        Args:
            x: BEV feature map tensor (B, C, H, W)
            
        Returns:
            Dictionary containing detection outputs:
            - cls_preds: Classification predictions
            - box_preds: Bounding box predictions
        """
        # List to store intermediate feature maps
        features = []
        
        # Process through conv blocks
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        # Generate detections from final feature map
        detections = self.detect_head(features[-1])
        
        return detections
    
class ConvBlock(nn.Module):
    """
    Basic convolutional block with batch normalization and ReLU activation.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 stride: int = 1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                     kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Add a second conv layer for better feature extraction
            nn.Conv2d(out_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class DetectionHead(nn.Module):
    """
    Detection head for 3D object detection.
    Predicts class scores and bounding box parameters.
    """
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 num_anchors: int):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification branch
        self.cls_head = nn.Conv2d(
            in_channels, num_anchors * num_classes,
            kernel_size=1, stride=1, padding=0
        )
        
        # Box regression branch
        # For each anchor: (x, y, z, w, l, h, θ)
        self.box_head = nn.Conv2d(
            in_channels, num_anchors * 7,
            kernel_size=1, stride=1, padding=0
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate detection predictions.
        
        Args:
            x: Input feature map (B, C, H, W)
            
        Returns:
            Dictionary containing:
            - cls_preds: Class predictions (B, num_anchors * num_classes, H, W)
            - box_preds: Box predictions (B, num_anchors * 7, H, W)
        """
        batch_size = x.shape[0]
        
        # Generate predictions
        cls_preds = self.cls_head(x)
        box_preds = self.box_head(x)
        
        return {
            'cls_preds': cls_preds,
            'box_preds': box_preds
        }