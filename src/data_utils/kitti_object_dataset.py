import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

class KITTIObjectDataset(Dataset):
    """
    Dataset class for KITTI 3D object detection,
    handling both point clouds and object labels.
    """
    def __init__(self, 
                 base_path: str,
                 split: str = 'training',
                 classes: List[str] = ['Car', 'Pedestrian', 'Cyclist']):
        """
        Initialize dataset.
        
        Args:
            base_path: Path to KITTI dataset
            split: 'training' or 'testing'
            classes: List of classes to detect
        """
        self.base_path = base_path
        self.split = split
        self.classes = {name: idx for idx, name in enumerate(classes)}
        
        # Get file paths
        self.lidar_dir = os.path.join(base_path, split, 'velodyne')
        self.label_dir = os.path.join(base_path, split, 'label_2')
        self.calib_dir = os.path.join(base_path, split, 'calib')
        
        # Get all frame IDs
        self.frame_ids = [f.split('.')[0] for f in os.listdir(self.lidar_dir)]
        
    def __len__(self) -> int:
        return len(self.frame_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get data for a single frame."""
        frame_id = self.frame_ids[idx]
        
        # Load point cloud
        points = self.load_point_cloud(frame_id)
        
        # Load calibration
        calib = self.load_calibration(frame_id)
        
        # Load labels (if in training split)
        if self.split == 'training':
            labels = self.load_labels(frame_id)
        else:
            labels = None
        
        return {
            'frame_id': frame_id,
            'points': points,
            'calib': calib,
            'labels': labels
        }
    
    def load_point_cloud(self, frame_id: str) -> np.ndarray:
        """Load LiDAR point cloud."""
        file_path = os.path.join(self.lidar_dir, f'{frame_id}.bin')
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    
    def load_calibration(self, frame_id: str) -> Dict:
        """Load calibration data."""
        calib_file = os.path.join(self.calib_dir, f'{frame_id}.txt')
        calib_data = {}
        
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                calib_data[key] = np.array([float(x) for x in value.split()])
        
        # Reshape matrices
        calib_data['P2'] = calib_data['P2'].reshape(3, 4)
        calib_data['Tr_velo_to_cam'] = calib_data['Tr_velo_to_cam'].reshape(3, 4)
        
        return calib_data
    
    def load_labels(self, frame_id: str) -> List[Dict]:
        """Load object labels."""
        label_file = os.path.join(self.label_dir, f'{frame_id}.txt')
        labels = []
        
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.split()
                class_name = parts[0]
                
                # Skip classes we're not interested in
                if class_name not in self.classes:
                    continue
                
                # Parse object data
                obj = {
                    'type': class_name,
                    'class_id': self.classes[class_name],
                    'truncation': float(parts[1]),
                    'occlusion': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox': [float(x) for x in parts[4:8]],  # 2D bbox
                    'dimensions': [float(x) for x in parts[8:11]],  # 3D size
                    'location': [float(x) for x in parts[11:14]],  # 3D location
                    'rotation_y': float(parts[14])  # Rotation
                }
                
                labels.append(obj)
        
        return labels