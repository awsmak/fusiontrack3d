import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, Tuple, Optional
import open3d as o3d

class DataVisualizer:
    """
    Visualization tools for multi-modal sensor data from KITTI dataset.
    Provides methods to visualize:
    1. Camera images
    2. LiDAR point clouds
    3. Projected LiDAR points on camera images
    4. Bird's eye view of LiDAR data
    """
    
    @staticmethod
    def visualize_frame(frame_data: Dict, show_lidar_overlay: bool = True) -> None:
        """
        Visualize a single frame with optional LiDAR overlay.
        
        Args:
            frame_data: Dictionary containing 'image', 'points', and 'calib'
            show_lidar_overlay: If True, project LiDAR points onto the image
        """
        image = frame_data['image'].copy()
        points = frame_data['points']
        calib = frame_data['calib']
        
        if show_lidar_overlay:
            # Convert points to homogeneous coordinates
            points_h = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
            
            # Transform LiDAR points to camera frame
            points_cam = np.dot(points_h, calib['Tr'].T)
            
            # Apply rectification
            points_rect = np.dot(points_cam, calib['R0_rect'].T)
            
            # Project to image plane
            points_proj = np.dot(points_rect, calib['P2'].T)
            pixels = points_proj[:, :2] / points_proj[:, 2:3]
            
            # Filter valid points
            mask = (points_rect[:, 2] > 0) & \
                   (pixels[:, 0] >= 0) & (pixels[:, 0] < image.shape[1]) & \
                   (pixels[:, 1] >= 0) & (pixels[:, 1] < image.shape[0])
            
            # Color points by depth
            depths = points_rect[mask, 2]
            pixels = pixels[mask].astype(np.int32)
            
            # Create color mapping based on depth
            colors = plt.cm.viridis((depths - depths.min()) / (depths.max() - depths.min()))
            colors = (colors[:, :3] * 255).astype(np.uint8)
            
            # Draw points on image
            for (x, y), color in zip(pixels, colors):
                cv2.circle(image, (x, y), 2, color.tolist(), -1)
        
        # Display result
        plt.figure(figsize=(15, 5))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def visualize_point_cloud(points: np.ndarray, 
                            view_dims: str = '3d') -> None:
        """
        Visualize LiDAR point cloud using Open3D.
        
        Args:
            points: Nx4 array of points (x, y, z, intensity)
            view_dims: '3d' or 'bev' (bird's eye view)
        """
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Color points by height
        colors = plt.cm.viridis((points[:, 2] - points[:, 2].min()) / 
                               (points[:, 2].max() - points[:, 2].min()))
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
        if view_dims == 'bev':
            # Set up for bird's eye view
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            
            # Set view for top-down perspective
            view_control = vis.get_view_control()
            view_control.set_zoom(0.7)
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, 1, 0])
            view_control.set_front([0, 0, 1])  # Looking down
            
            vis.run()
            vis.destroy_window()
        else:
            # Regular 3D visualization
            o3d.visualization.draw_geometries([pcd])
    
    @staticmethod
    def visualize_frame_multi_view(frame_data: Dict) -> None:
        """
        Show multiple visualizations of the same frame:
        1. Original image
        2. Image with LiDAR overlay
        3. Bird's eye view
        """
        plt.figure(figsize=(20, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(frame_data['image'])
        plt.title('Camera Image')
        plt.axis('off')
        
        # Image with LiDAR overlay
        plt.subplot(132)
        image = frame_data['image'].copy()
        points = frame_data['points']
        calib = frame_data['calib']
        
        # Project LiDAR points onto image
        points_h = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
        points_cam = np.dot(points_h, calib['Tr'].T)
        points_rect = np.dot(points_cam, calib['R0_rect'].T)
        points_proj = np.dot(points_rect, calib['P2'].T)
        pixels = points_proj[:, :2] / points_proj[:, 2:3]
        
        mask = (points_rect[:, 2] > 0) & \
            (pixels[:, 0] >= 0) & (pixels[:, 0] < image.shape[1]) & \
            (pixels[:, 1] >= 0) & (pixels[:, 1] < image.shape[0])
        
        depths = points_rect[mask, 2]
        pixels = pixels[mask].astype(np.int32)
        colors = plt.cm.viridis((depths - depths.min()) / (depths.max() - depths.min()))
        colors = (colors[:, :3] * 255).astype(np.uint8)
        
        for (x, y), color in zip(pixels, colors):
            cv2.circle(image, (x, y), 2, color.tolist(), -1)
        
        plt.imshow(image)
        plt.title('LiDAR Projection')
        plt.axis('off')
        
        # Bird's eye view
        plt.subplot(133)
        # Filter points within range
        side_range = (-20, 20)
        size_pixels = 800
        mask = (points[:, 0] >= side_range[0]) & (points[:, 0] <= side_range[1]) & \
            (points[:, 1] >= side_range[0]) & (points[:, 1] <= side_range[1])
        points_bev = points[mask]
        
        # Create BEV image
        resolution = (side_range[1] - side_range[0]) / size_pixels
        bev_image = np.zeros((size_pixels, size_pixels, 3), dtype=np.uint8)
        
        # Convert points to pixel coordinates
        x_img = ((points_bev[:, 0] - side_range[0]) / resolution).astype(np.int32)
        y_img = ((points_bev[:, 1] - side_range[0]) / resolution).astype(np.int32)
        
        # Color by height
        colors = plt.cm.viridis((points_bev[:, 2] - points_bev[:, 2].min()) / 
                            (points_bev[:, 2].max() - points_bev[:, 2].min()))
        colors = (colors[:, :3] * 255).astype(np.uint8)
        
        # Plot points
        for x, y, color in zip(x_img, y_img, colors):
            if 0 <= x < size_pixels and 0 <= y < size_pixels:
                bev_image[y, x] = color
        
        plt.imshow(bev_image)
        plt.title('Bird\'s Eye View')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()