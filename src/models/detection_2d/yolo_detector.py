from typing import List, Dict, Tuple
import numpy as np
from ultralytics import YOLO
import torch
import cv2

class YOLODetector:
    """
    2D object detector using YOLOv8 for camera images.
    """
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
        results = self.model(image, conf=conf_threshold)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = results.names[class_id]
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            })
            
        return detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            class_name = det['class_name']
            
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return img
