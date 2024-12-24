import torch
from typing import Dict, List, Tuple
import numpy as np

def prepare_targets(batch_data: Dict[str, torch.Tensor],
                   anchors: torch.Tensor,
                   config: 'TrainingConfig') -> Dict[str, torch.Tensor]:
    """
    Prepare target tensors for training.
    
    Args:
        batch_data: Dictionary containing ground truth data
        anchors: Anchor boxes
        config: Training configuration
    
    Returns:
        Dictionary containing target tensors
    """
    cls_targets = []
    box_targets = []
    cls_weights = []
    box_weights = []
    
    for batch_idx in range(len(batch_data['labels'])):
        # Get ground truth boxes for this sample
        gt_boxes = batch_data['labels'][batch_idx]
        
        # Match anchors to ground truth boxes
        matches = assign_targets_to_anchors(anchors, gt_boxes, config)
        
        # Generate targets based on matches
        sample_cls_targets, sample_box_targets, \
        sample_cls_weights, sample_box_weights = generate_targets(
            matches, anchors, gt_boxes, config
        )
        
        cls_targets.append(sample_cls_targets)
        box_targets.append(sample_box_targets)
        cls_weights.append(sample_cls_weights)
        box_weights.append(sample_box_weights)
    
    return {
        'cls_targets': torch.stack(cls_targets),
        'box_targets': torch.stack(box_targets),
        'cls_weights': torch.stack(cls_weights),
        'box_weights': torch.stack(box_weights)
    }

def assign_targets_to_anchors(anchors: torch.Tensor,
                            gt_boxes: List[Dict],
                            config: 'TrainingConfig') -> List[Dict]:
    """
    Match anchors to ground truth boxes using IoU.
    """
    matches = []
    iou_threshold = 0.5
    
    # Convert ground truth boxes to tensor
    gt_boxes_tensor = boxes_to_tensor(gt_boxes)
    
    # Calculate IoU between anchors and ground truth boxes
    ious = calculate_3d_iou(anchors, gt_boxes_tensor)
    
    # Assign matches based on IoU
    max_ious, gt_indices = ious.max(dim=1)
    
    # Positive matches: IoU > threshold
    pos_mask = max_ious > iou_threshold
    
    # Create match information
    for anchor_idx in range(len(anchors)):
        if pos_mask[anchor_idx]:
            gt_idx = gt_indices[anchor_idx]
            matches.append({
                'anchor_idx': anchor_idx,
                'gt_idx': gt_idx.item(),
                'iou': max_ious[anchor_idx].item()
            })
    
    return matches

def generate_targets(matches: List[Dict],
                    anchors: torch.Tensor,
                    gt_boxes: List[Dict],
                    config: 'TrainingConfig') -> Tuple[torch.Tensor, ...]:
    """
    Generate classification and regression targets based on matches.
    """
    num_anchors = len(anchors)
    num_classes = len(config.classes)
    
    # Initialize targets
    cls_targets = torch.zeros((num_anchors, num_classes))
    box_targets = torch.zeros((num_anchors, 7))  # 7 for (x,y,z,w,l,h,θ)
    cls_weights = torch.ones(num_anchors) * 0.1  # Background weight
    box_weights = torch.zeros(num_anchors)
    
    # Assign targets for positive matches
    for match in matches:
        anchor_idx = match['anchor_idx']
        gt_idx = match['gt_idx']
        
        # Classification target (one-hot)
        cls_targets[anchor_idx][gt_boxes[gt_idx]['class_id']] = 1
        cls_weights[anchor_idx] = 1.0
        
        # Box regression target
        box_targets[anchor_idx] = compute_box_targets(
            anchors[anchor_idx],
            gt_boxes[gt_idx]
        )
        box_weights[anchor_idx] = 1.0
    
    return cls_targets, box_targets, cls_weights, box_weights

def compute_box_targets(anchor: torch.Tensor,
                       gt_box: Dict) -> torch.Tensor:
    """
    Compute box regression targets.
    """
    # Convert ground truth box parameters to tensor
    gt_center = torch.tensor(gt_box['location'])
    gt_size = torch.tensor(gt_box['dimensions'])
    gt_angle = torch.tensor(gt_box['rotation_y'])
    
    # Extract anchor parameters
    anchor_center = anchor[:3]
    anchor_size = anchor[3:6]
    anchor_angle = anchor[6]
    
    # Compute targets
    center_targets = (gt_center - anchor_center) / anchor_size
    size_targets = torch.log(gt_size / anchor_size)
    angle_target = gt_angle - anchor_angle
    
    return torch.cat([center_targets, size_targets, angle_target.unsqueeze(0)])