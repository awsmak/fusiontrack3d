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
                 date: str = '2011_09_26',     # Add date parameter
                 drive: str = '0001',          # Add drive parameter
                 classes: List[str] = ['Car', 'Pedestrian', 'Cyclist']):
        """
        Initialize dataset.
        
        Args:
            base_path: Path to KITTI dataset root
            date: Date of the sequence (e.g., '2011_09_26')
            drive: Drive number (e.g., '0001')
            classes: List of classes to detect
        """
        self.base_path = base_path
        self.date = date
        self.drive = drive
        self.classes = {name: idx for idx, name in enumerate(classes)}
        
        # Get paths for raw data
        self.sequence_path = os.path.join(
            base_path, 'raw', 
            date, 
            f"{date}_drive_{drive}_sync"
        )
        
        self.lidar_dir = os.path.join(self.sequence_path, 'velodyne_points/data')
        self.calib_path = os.path.join(base_path, 'raw', date)
        
        # Get all frame IDs
        self.frame_ids = sorted([
            f.split('.')[0] 
            for f in os.listdir(self.lidar_dir) 
            if f.endswith('.bin')
        ])
        
    def __len__(self) -> int:
        return len(self.frame_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get data for a single frame."""
        frame_id = self.frame_ids[idx]
        
        # Load point cloud
        points = self.load_point_cloud(frame_id)
        
        # Load calibration
        calib = self.load_calibration()
        
        # For raw data, we don't have labels
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
    
    def load_calibration(self) -> Dict:
        """Load calibration data."""
        calib_data = {}
        
        # Load velo to cam calibration
        calib_file = os.path.join(self.calib_path, 'calib_velo_to_cam.txt')
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    # Skip lines that don't contain calibration data
                    if 'R' in key or 'T' in key:  # Only process R and T matrices
                        calib_data[key] = np.array([float(x) for x in value.strip().split()])
        
        # Load camera calibration
        cam_calib_file = os.path.join(self.calib_path, 'calib_cam_to_cam.txt')
        with open(cam_calib_file, 'r') as f:
            for line in f.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    # Only process projection matrix P2
                    if key.strip() == 'P2':
                        calib_data[key] = np.array([float(x) for x in value.strip().split()]).reshape(3, 4)
        
        # Process calibration data
        if 'R' in calib_data and 'T' in calib_data:
            # Create transformation matrix
            R = calib_data['R'].reshape(3, 3)
            T = calib_data['T'].reshape(3, 1)
            velo_to_cam = np.vstack((np.hstack([R, T]), np.array([0., 0., 0., 1.])))
            calib_data['velo_to_cam'] = velo_to_cam
        
        return calib_data