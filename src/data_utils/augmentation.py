import numpy as np
import torch
from typing import Dict, Tuple, List
import copy
import random

class PointCloudAugmentor:
    """
    Augmentation class for point cloud data and 3D bounding boxes.
    Implements various augmentation techniques specific to 3D object detection.
    """
    def __init__(self, 
                 rotation_range: Tuple[float, float] = (-np.pi/4, np.pi/4),
                 scaling_range: Tuple[float, float] = (0.95, 1.05),
                 translation_range: Tuple[float, float] = (-5, 5),
                 flip_probability: float = 0.5):
        """
        Initialize augmentation parameters.
        
        Args:
            rotation_range: Range for random rotation (min, max) in radians
            scaling_range: Range for random scaling (min, max)
            translation_range: Range for random translation (min, max) in meters
            flip_probability: Probability of random flip along x-axis
        """
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.translation_range = translation_range
        self.flip_probability = flip_probability
    
    def augment(self, 
                points: np.ndarray, 
                boxes: List[Dict] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply augmentation to point cloud and boxes.
        
        Args:
            points: (N, 4) array of points (x, y, z, intensity)
            boxes: List of box dictionaries with location, dimensions, rotation
            
        Returns:
            Augmented points and boxes
        """
        # Make copies to avoid modifying original data
        points = points.copy()
        if boxes is not None:
            boxes = copy.deepcopy(boxes)
        
        # Random rotation
        if self.rotation_range is not None:
            angle = np.random.uniform(*self.rotation_range)
            points, boxes = self.rotate(points, boxes, angle)
        
        # Random scaling
        if self.scaling_range is not None:
            scale = np.random.uniform(*self.scaling_range)
            points, boxes = self.scale(points, boxes, scale)
        
        # Random translation
        if self.translation_range is not None:
            translation = np.random.uniform(
                self.translation_range[0],
                self.translation_range[1],
                size=3
            )
            points, boxes = self.translate(points, boxes, translation)
        
        # Random flip
        if np.random.random() < self.flip_probability:
            points, boxes = self.flip_x(points, boxes)
        
        return points, boxes
    
    @staticmethod
    def rotate(points: np.ndarray, 
              boxes: List[Dict], 
              angle: float) -> Tuple[np.ndarray, List[Dict]]:
        """Rotate points and boxes around z-axis."""
        # Create rotation matrix
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        R = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        
        # Rotate points
        points[:, :3] = points[:, :3] @ R.T
        
        # Rotate boxes
        if boxes is not None:
            for box in boxes:
                # Rotate center location
                box['location'][:2] = box['location'][:2] @ R[:2, :2].T
                # Update rotation angle
                box['rotation_y'] = (box['rotation_y'] + angle) % (2 * np.pi)
        
        return points, boxes
    
    @staticmethod
    def scale(points: np.ndarray, 
             boxes: List[Dict], 
             scale: float) -> Tuple[np.ndarray, List[Dict]]:
        """Scale points and boxes."""
        # Scale points
        points[:, :3] *= scale
        
        # Scale boxes
        if boxes is not None:
            for box in boxes:
                box['location'] *= scale
                box['dimensions'] *= scale
        
        return points, boxes
    
    @staticmethod
    def translate(points: np.ndarray,
                 boxes: List[Dict],
                 translation: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Translate points and boxes."""
        # Translate points
        points[:, :3] += translation
        
        # Translate boxes
        if boxes is not None:
            for box in boxes:
                box['location'] += translation
        
        return points, boxes
    
    @staticmethod
    def flip_x(points: np.ndarray,
              boxes: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Flip points and boxes along x-axis."""
        # Flip points
        points[:, 0] = -points[:, 0]
        
        # Flip boxes
        if boxes is not None:
            for box in boxes:
                box['location'][0] = -box['location'][0]
                box['rotation_y'] = np.pi - box['rotation_y']
        
        return points, boxes
    
    def apply_noise(self, points: np.ndarray, 
                   std: float = 0.02) -> np.ndarray:
        """Add random noise to point coordinates."""
        noise = np.random.normal(0, std, size=points[:, :3].shape)
        points[:, :3] += noise
        return points
    
    def random_dropout(self, points: np.ndarray, 
                      dropout_prob: float = 0.05) -> np.ndarray:
        """Randomly dropout points."""
        mask = np.random.random(size=len(points)) > dropout_prob
        return points[mask]