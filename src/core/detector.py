from ultralytics import YOLO
import yaml
from pathlib import Path


class PersonDetector:
    """YOLO ile insan tespiti"""
    
    def __init__(self, config_path="configs/model.yaml"):
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.config = config['detection']
        self.model = YOLO(self.config['model_name'])
        self.conf_thresh = self.config['confidence_threshold']
        self.iou_thresh = self.config['iou_threshold']
        self.device = self.config['device']
        
    def detect(self, frame):
        """Frame üzerinde detection
        
        Returns:
            boxes: [[x1, y1, x2, y2, conf], ...]
        """
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            classes=[0],  # sadece insan sınıfı
            verbose=False
        )[0]
        
        boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                boxes.append([x1, y1, x2, y2, conf])
        
        return boxes
