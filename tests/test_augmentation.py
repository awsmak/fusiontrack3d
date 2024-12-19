import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.data_utils.kitti_object_dataset import KITTIObjectDataset
from src.data_utils.augmentation import PointCloudAugmentor

def visualize_augmentation(points_orig, points_aug, title="Point Cloud Comparison"):
    """Visualize original and augmented point clouds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot original points
    ax1.scatter(points_orig[:, 0], points_orig[:, 1], 
                c=points_orig[:, 2], cmap='viridis', s=1)
    ax1.set_title('Original Point Cloud')
    ax1.set_aspect('equal')
    
    # Plot augmented points
    ax2.scatter(points_aug[:, 0], points_aug[:, 1], 
                c=points_aug[:, 2], cmap='viridis', s=1)
    ax2.set_title('Augmented Point Cloud')
    ax2.set_aspect('equal')
    
    plt.suptitle(title)
    plt.show()

def test_augmentation():
    # Initialize dataset with correct path and sequence
    dataset = KITTIObjectDataset(
        base_path='/workspace/data/kitti',
        date='2011_09_26',
        drive='0001'
    )
    # Initialize augmentor
    augmentor = PointCloudAugmentor(
        rotation_range=(-np.pi/4, np.pi/4),
        scaling_range=(0.95, 1.05),
        translation_range=(-5, 5),
        flip_probability=0.5
    )
    
    # Load a sample
    sample = dataset[0]
    points = sample['points']
    boxes = sample['labels'] if 'labels' in sample else None
    
    # Apply augmentation
    points_aug, boxes_aug = augmentor.augment(points, boxes)
    
    # Visualize results
    visualize_augmentation(points, points_aug, "Augmentation Result")
    
    # Print statistics
    print("\nPoint Cloud Statistics:")
    print(f"Original points: {len(points)}")
    print(f"Augmented points: {len(points_aug)}")
    
    if boxes is not None:
        print("\nBounding Box Statistics:")
        print(f"Original boxes: {len(boxes)}")
        print(f"Augmented boxes: {len(boxes_aug)}")

if __name__ == "__main__":
    test_augmentation()