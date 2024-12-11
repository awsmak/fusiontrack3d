import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # Insert at beginning of path
print(f"Python path: {sys.path}")  # Debug print

import matplotlib.pyplot as plt
from src.data_utils.kitti_dataset import KITTIDataset
from src.models.detection_2d import YOLODetector  # Modified import statement

def test_2d_detection():
    # Initialize dataset
    dataset = KITTIDataset(
        base_path='/workspace/data/kitti/raw',
        date='2011_09_26',
        drive='0001'
    )
    
    # Initialize detector
    detector = YOLODetector()
    
    # Load a frame
    frame_data = dataset.get_frame_data(0)
    image = frame_data['image']
    
    # Perform detection
    detections = detector.detect(image)
    
    # Visualize results
    image_with_dets = detector.visualize_detections(image, detections)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(image_with_dets)
    plt.axis('off')
    plt.title('YOLOv8 Detections')
    plt.show()
    
    # Print detection results
    print("\nDetections:")
    for det in detections:
        print(f"Class: {det['class_name']}, Confidence: {det['confidence']:.2f}")

if __name__ == "__main__":
    test_2d_detection()