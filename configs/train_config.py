from typing import Dict, Any

class TrainingConfig:
    """Configuration for training the PointPillars network."""
    
    def __init__(self):
        # Dataset parameters
        self.data_path = '/workspace/data/kitti'
        self.classes = ['Car', 'Pedestrian', 'Cyclist']
        
        # Model parameters
        self.point_cloud_range = (-50, -50, -3, 50, 50, 1)  # x_min, y_min, z_min, x_max, y_max, z_max
        self.voxel_size = (0.16, 0.16, 4)     # Size of each pillar
        self.max_points_per_pillar = 100       # Maximum number of points in each pillar
        self.max_pillars = 12000               # Maximum number of pillars
        
        # Training parameters
        self.batch_size = 2
        self.num_epochs = 80
        self.learning_rate = 0.001
        self.weight_decay = 0.01
        
        # Loss weights
        self.cls_weight = 1.0
        self.reg_weight = 2.0
        self.dir_weight = 0.2
        
        # Anchor parameters
        self.anchor_sizes = {
            'Car': [(4.7, 2.1, 1.7)],
            'Pedestrian': [(0.8, 0.6, 1.7)],
            'Cyclist': [(1.8, 0.6, 1.7)]
        }
        self.anchor_rotations = [0, 1.57]  # 0 and 90 degrees
        
        # Data augmentation
        self.use_augmentation = True
        self.global_rotation_noise = (-0.785, 0.785)
        self.global_scaling_noise = (0.95, 1.05)