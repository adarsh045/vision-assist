from visionassist.model.yolo import YOLOModel
from visionassist.logger import logger
import cv2

model = YOLOModel()

def test_yolo_model_initialization():
    assert model.model is not None

def test_yolo_model_detection():
    frame = cv2.imread("tests/assets/truck.jpg")
    detections = model.detect(frame)
    assert isinstance(detections, list)

def test_yolo_model_save_with_bbox():
    frame = cv2.imread("tests/assets/trucks.jpg")
    detections = model.detect(frame)
    file_path = model.save_with_bbox(frame, detections)
    assert isinstance(file_path, str)

if __name__ == "__main__":
    test_yolo_model_initialization()
    test_yolo_model_detection()
    test_yolo_model_save_with_bbox()