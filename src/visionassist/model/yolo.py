import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from visionassist.config import ALLOWED_LABELS, MIN_CONFIDENCE, IMAGE_DIR, YOLO_MODEL_NAME
from visionassist.model.color import get_random_color
from visionassist.logger import logger

class YOLOModel:
    def __init__(self):
        logger.info(f"Initializing YOLO model : {YOLO_MODEL_NAME}")
        self.model = YOLO(YOLO_MODEL_NAME)
        self.class_ids =  [class_info[0] for class_info in self.model.names.items() if class_info[1].lower() in ALLOWED_LABELS]
        os.makedirs(IMAGE_DIR, exist_ok=True)
        logger.info(f"YOLO model initialized with allowed class ids: {self.class_ids}")

    def detect(self, frame:np.ndarray):
        """Run YOLO detection and filter by allowed class ids + confidence."""
        results = self.model(frame, classes=self.class_ids, verbose=False)[0]

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < MIN_CONFIDENCE:
                continue

            label = results.names[cls_id]
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
            })

        return detections

    def save_with_bbox(self, frame, detections):
        """Save cropped object image."""
        if not detections:
            return ""
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            cv2.rectangle(frame, (x1, y1), (x2, y2), get_random_color(), 2)
            cv2.putText(frame, f"{label} {det['confidence']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
        timestamp = int(time.time())
        filename = f"{label}_{timestamp}.jpg"
        filepath = os.path.join(IMAGE_DIR, filename)

        cv2.imwrite(filepath, frame)

        return filepath
