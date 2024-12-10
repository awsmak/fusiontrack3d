import os
import numpy as np
import cv2
import pykitti
from typing import Dict, Tuple, Optional, List

class KITTIDataset:
    """
    Data loader for the KITTI dataset. Handles both camera images and LiDAR point clouds.
    """

    def __init__(self, base_path: str, date: str, drive: str):
        """
        Initialize the KITTI dataset loader.
        
        Args:
            base_path: Root dir of KITTI dataset (e.g., 'workspace/data/kitti/raw')
            date: Date of the sequence (e.g., '2011_09_26')
            drive: Drive number (e.g., '0001')
        """
        self.base_path = base_path
        self.date = date
        self.drive = drive
        
        # Construct paths
        self.sequence_path = os.path.join(
            base_path,
            date,
            f"{date}_drive_{drive}_sync"
        )
        self.calib_path = os.path.join(base_path, date)  # Add this line

        if not self._verify_paths():
            raise FileNotFoundError(f"Dataset paths not found in {self.sequence_path}")
        
        # Load calibration data
        self.calib = self._load_calibration()
        
        # Initialize frame indices
        self.frame_indices = self._get_available_frames()

    def _verify_paths(self) -> bool:
        """Verify all required dataset paths exist."""
        required_paths = [
            self.sequence_path,
            os.path.join(self.sequence_path, 'image_02', 'data'),  # Added 'data' subdirectory
            os.path.join(self.sequence_path, 'velodyne_points', 'data'),  # Added 'data' subdirectory
            os.path.join(self.calib_path, 'calib_cam_to_cam.txt'),
            os.path.join(self.calib_path, 'calib_velo_to_cam.txt')
        ]
        return all(os.path.exists(path) for path in required_paths)

    def _read_calib_file(self, filepath: str) -> Dict:
        """
        Read and parse calibration files.
        
        Args:
            filepath: Path to calibration file

        Returns:
            Dict containing calibration matrices    
        """
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                if ':' in line:  # Only process valid key-value pairs
                    key, value = line.split(':', 1)
                    key = key.strip()
                    
                    # Skip non-matrix data
                    if key == 'calib_time':
                        continue
                        
                    # Convert string values to numpy arrays
                    try:
                        # Handle corner cases for different matrix sizes
                        values = [float(x) for x in value.strip().split()]
                        
                        # Determine matrix shape based on the key prefix
                        if key.startswith('K') or key.startswith('R'):  # 3x3 matrices
                            data[key] = np.array(values).reshape(3, 3)
                        elif key.startswith('P'):  # 3x4 projection matrices
                            data[key] = np.array(values).reshape(3, 4)
                        elif key.startswith('T'):  # 3x1 translation vectors
                            data[key] = np.array(values).reshape(3, 1)
                        elif key.startswith('S'):  # 2x1 size vectors
                            data[key] = np.array(values)
                        elif key.startswith('D'):  # Distortion parameters
                            data[key] = np.array(values)
                        else:
                            data[key] = np.array(values)
                    except ValueError:
                        continue
                    except IndexError:
                        continue
        return data

    def _load_calibration(self) -> Dict:
        """
        Load and process all calibration files.
        
        Returns:
            Dictionary containing processed calibration matrices:
            - P2: Projection matrix for left RGB camera (camera 02)
            - Tr: Transform from velodyne to reference camera coordinates
            - R0_rect: Rectification matrix for reference camera
        """
        velo_to_cam = self._read_calib_file(
            os.path.join(self.calib_path, 'calib_velo_to_cam.txt')
        )
        cam_to_cam = self._read_calib_file(
            os.path.join(self.calib_path, 'calib_cam_to_cam.txt')
        )
        
        calib_data = {}
        
        # Get P2 (left color camera) projection matrix
        calib_data['P2'] = cam_to_cam['P_rect_02']  # Already 3x4
        
        # Get R0_rect (rectification matrix for reference camera)
        R0_rect = cam_to_cam['R_rect_00']  # 3x3
        # Convert to 4x4 for easier multiplication
        R0_rect_4x4 = np.eye(4)
        R0_rect_4x4[:3, :3] = R0_rect
        calib_data['R0_rect'] = R0_rect_4x4
        
        # Get Tr_velo_to_cam (transform from velodyne to camera)
        # Create 4x4 transform matrix
        Tr_velo_to_cam = np.eye(4)
        # Rotation matrix (3x3)
        Tr_velo_to_cam[:3, :3] = velo_to_cam['R']
        # Translation vector (3x1)
        Tr_velo_to_cam[:3, 3] = velo_to_cam['T'].flatten()
        calib_data['Tr'] = Tr_velo_to_cam
        
        # Additional calibration data that might be useful
        calib_data['K2'] = cam_to_cam['K_02']  # Intrinsic matrix for left color camera
        calib_data['D2'] = cam_to_cam['D_02']  # Distortion coefficients
        
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

    def get_frame_data(self, frame_idx: int) -> Dict:
        """
        Load data for specific frame, including camera image and LiDAR points.

        Args:
            frame_idx: Index of frame to load

        Returns:
            Dictionary containing:
                - image: RGB camera image (H,W,3)
                - points: LiDAR point cloud (N, 4) - x,y,z, intensity
                - calib: Calibration data for the frame
        """
        if frame_idx not in self.frame_indices:
            raise ValueError(f"Frame index {frame_idx} not available")
        
        # Load camera image
        image_path = os.path.join(
            self.sequence_path,
            'image_02',
            'data',
            f'{frame_idx:010d}.png'
        )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load LiDAR points
        points_path = os.path.join(
            self.sequence_path,
            'velodyne_points',
            'data',
            f'{frame_idx:010d}.bin'
        )
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)

        return {
            'image': image,
            'points': points,
            'calib': self.calib,
            'frame_idx': frame_idx
        }