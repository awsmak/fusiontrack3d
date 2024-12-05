import os
import numpy as np
import cv2
import pykitti
from typing import Dict, Tuple, Optional, List

class KITTIDataset:
    """
    Data loader for the KITTI dataset. handles both camera images and LiDAR point clouds.
    """

    def __init__(self, base_path: str, date: str, drive: str):
        """
        initialize the kitti dataset loader.
        
        args:
            base_path: Root dir of kitti dataset(e.g 'workspace/data/kitti/raw')
            date: Date of the sequence (e.g. '2011_09_26')
            drive: Drive number (e.g. '0001')

        The raw KITTI dataset has this structure:
        base_path/
        ├── date/
        │   ├── calib_cam_to_cam.txt
        │   ├── calib_velo_to_cam.txt
        │   └── date_drive_XXXX_sync/
        │       ├── image_02/        # Left RGB camera images
        │       └── velodyne_points/ # LiDAR point clouds

        """

        self.base_path = base_path
        self.date = date
        self.drive = drive

        #construct the sequnce path
        self.sequence_path = os.path.join(
            base_path,
            date,
            f"{date}_drive{drive}_sync"
        )

        if not self._verify_paths():
            raise FileNotFoundError("Dataset paths not found in {self.sequence}_path")
        
        #load caib data
        self.calib = self._load_calibration()

        #initialize frame indices
        self.frame_indices = self._get_available_frames()

    def _verify_paths(self) -> bool:
        """Verify all dataset paths"""
        required_paths = [
            self.sequence_path,
            os.path.join(self.sequence_path, 'image_02'),
            os.path.join(self.sequence_path, 'velodyne_points'),
            os.path.join(self.base_path, self.date) # fpr calib files
        ]
        return all(os.path.exists(path)for path in required_paths)
    
    def _read_caib_file(self, filepath: str) ->Dict:
        """
        Read and parse calibration files.
        
        Args:
            filepath: Path to calibration file

        Returns:
            Dict containg calibration matrices    
        """
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                if ":" in line: # onlu process calid key-value pair
                    key, value  = line.split(':', 1)
                    try:
                        #convert string values to numpy arrays
                        data[key.strip()] = np.array([float(x) for x in value.split()])
                    except ValueError:
                        #skip lines that cant be converted to floats
                        continue
        return data
    

    
    def _load_calibration(self) -> Dict:
        """
        Load and process all calibration files.
        
        Returns:
            Dictionary containing processed calibration matrices:
            - P2: Projection matrix for left RGB camera
            - Tr: Transform from velodyne to reference camera coordinates
            - R0_rect: Rectification matrix for reference camera
        """
        # Load individual calibration files
        velo_to_cam = self._read_calib_file(
            os.path.join(self.calib_path, 'calib_velo_to_cam.txt')
        )
        cam_to_cam = self._read_calib_file(
            os.path.join(self.calib_path, 'calib_cam_to_cam.txt')
        )
        
        # Process calibration data
        calib_data = {}
        
        # Camera projection matrix (3x4)
        calib_data['P2'] = cam_to_cam['P2'].reshape(3, 4)
        
        # Rectification matrix (3x3)
        R0_rect = cam_to_cam['R_rect_00'].reshape(3, 3)
        calib_data['R0_rect'] = np.eye(4)  # Convert to 4x4
        calib_data['R0_rect'][:3, :3] = R0_rect
        
        # Velodyne to camera transform (4x4)
        Tr_velo_to_cam = np.eye(4)
        r = velo_to_cam['R'].reshape(3, 3)
        t = velo_to_cam['T'].reshape(3, 1)
        Tr_velo_to_cam[:3, :3] = r
        Tr_velo_to_cam[:3, 3] = t.flatten()
        calib_data['Tr'] = Tr_velo_to_cam
        
        return calib_data
    
    def _get_available_frames(self) -> List[int]:
        """
        Get list of available frame indices in the sequence.

        Returns:
            List of frame indices available in both images and LiDAR directories
        """
        image_dir = os.path.join(self.sequence_path, 'image_02', 'data')
        frame_indices = []

        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith('.png'):
                frame_idx = int(filename.split('.')[0])
                frame_indices.append(frame_idx)

        return frame_indices
    
    def get_frame_data(self, frame_idx: int) ->Dict:
        """
        Load data for specific frame, including camera image, LiDAR points.

        Args:
            frame_idx: Index of frame to load

        Returns:
            Dictionary containing:
                - image: RGB camera (H,W,3)
                - points: LiDAR point cloud (N, 4) - x,y,z, intensity
                - calib: Calibration data for the frame
        """
        if frame_idx not in self.frame_indices:
            raise ValueError(f"Frame index {frame_idx} not available")
        
        #load camera image
        image_path = os.path.join(
            self.sequence_path,
            'image_02',
            'data',
            f'{frame_idx:010d}.png'

        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #load lidar points
        points_path = os.path.join(
            self.sequence_path,
            'velodyne_points',
            'data',
            f'{frame_idx:010d}.bin'
        )

        points = np.fromfile(points_path, dtype=np.float32.reshape(-1,4))

        return {
            'image': image,
            'points': points,
            'calib': self.calib,
            'frame_idx': frame_idx

        }