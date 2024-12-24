import numpy as np
from typing import List, Dict, Tuple
import torch

class DetectionEvaluator:
    """
    Evaluator class for 3D object detection.
    Implements metrics like Average Precision and Intersection over Union.
    """
    def __init__(self, iou_thresholds: List[float] = [0.5, 0.7]):
        self.iou_thresholds = iou_thresholds
    
    def evaluate(self, 
                predictions: List[Dict], 
                targets: List[Dict]) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of predicted boxes
            targets: List of ground truth boxes
            
        Returns:
            Dictionary containing metrics
        """
        metrics = {}
        
        # Calculate metrics for each IoU threshold
        for iou_thresh in self.iou_thresholds:
            ap = self.calculate_average_precision(
                predictions, targets, iou_thresh
            )
            metrics[f'AP@{iou_thresh}'] = ap
        
        return metrics
    
    @staticmethod
    def calculate_3d_iou(box1: np.ndarray, 
                        box2: np.ndarray) -> float:
        """
        Calculate 3D IoU between boxes.
        
        Args:
            box1, box2: Arrays containing [x, y, z, l, w, h, theta]
        """
        # Convert boxes to corners
        corners1 = DetectionEvaluator.box_to_corners(box1)
        corners2 = DetectionEvaluator.box_to_corners(box2)
        
        # Calculate intersection volume
        intersection = DetectionEvaluator.get_intersection_volume(
            corners1, corners2
        )
        
        # Calculate volumes
        vol1 = box1[3] * box1[4] * box1[5]  # l * w * h
        vol2 = box2[3] * box2[4] * box2[5]
        
        # Calculate IoU
        union = vol1 + vol2 - intersection
        iou = intersection / (union + 1e-16)
        
        return iou
    
    @staticmethod
    def box_to_corners(box: np.ndarray) -> np.ndarray:
        """Convert box parameters to corners."""
        x, y, z, l, w, h, theta = box
        
        # Create corner coordinates in reference frame
        corners = np.array([
            [l/2, w/2, h/2],
            [l/2, w/2, -h/2],
            [l/2, -w/2, h/2],
            [l/2, -w/2, -h/2],
            [-l/2, w/2, h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [-l/2, -w/2, -h/2]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Rotate corners
        corners = corners @ R.T
        
        # Translate corners
        corners = corners + np.array([x, y, z])
        
        return corners
    
    @staticmethod
    def get_intersection_volume(corners1: np.ndarray, 
                              corners2: np.ndarray) -> float:
        """Calculate intersection volume of two boxes."""
        # Project boxes onto XY, YZ, and XZ planes
        xy1 = corners1[:, :2]
        xy2 = corners2[:, :2]
        
        # Get min/max coordinates
        min1 = np.min(xy1, axis=0)
        max1 = np.max(xy1, axis=0)
        min2 = np.min(xy2, axis=0)
        max2 = np.max(xy2, axis=0)
        
        # Calculate intersection area
        intersection_xy = np.maximum(0, np.minimum(max1[0], max2[0]) - 
                                     np.maximum(min1[0], min2[0])) * \
                         np.maximum(0, np.minimum(max1[1], max2[1]) - 
                                     np.maximum(min1[1], min2[1]))
        
        # Calculate height intersection
        z_min1 = np.min(corners1[:, 2])
        z_max1 = np.max(corners1[:, 2])
        z_min2 = np.min(corners2[:, 2])
        z_max2 = np.max(corners2[:, 2])
        
        intersection_h = np.maximum(0, np.minimum(z_max1, z_max2) - 
                                    np.maximum(z_min1, z_min2))
        
        return intersection_xy * intersection_h
    
    def calculate_average_precision(self,
                                  predictions: List[Dict],
                                  targets: List[Dict],
                                  iou_threshold: float) -> float:
        """Calculate Average Precision for given IoU threshold."""
        # Sort predictions by confidence
        predictions = sorted(predictions, 
                           key=lambda x: x['confidence'], 
                           reverse=True)
        
        # Initialize arrays for precision-recall calculation
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        num_targets = len(targets)
        if num_targets == 0:
            return 0.0
        
        # Match predictions to targets
        matched_targets = []
        
        for i, pred in enumerate(predictions):
            best_iou = 0
            best_target_idx = -1
            
            # Find best matching target
            for j, target in enumerate(targets):
                if j in matched_targets:
                    continue
                    
                iou = self.calculate_3d_iou(
                    np.array(pred['box']),
                    np.array(target['box'])
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j
            
            # Check if match is good enough
            if best_iou >= iou_threshold:
                tp[i] = 1
                matched_targets.append(best_target_idx)
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_targets
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.0
            
        return ap