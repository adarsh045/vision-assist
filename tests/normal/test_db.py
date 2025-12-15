from visionassist.memory.database import Database
from visionassist.config import MAX_OBJECT_ENTRIES

db = Database()

def generate_test_data(n_images:50):
    data = []
    for i in range(n_images):
        image_path = f"test_image_{i}.jpg"
        objects = [
            {"label": "backpack", "confidence": 0.95, "bbox": (10, 20, 30, 40)},
            {"label": "umbrella", "confidence": 0.90, "bbox": (50, 60, 70, 80)},
            {"label": "handbag", "confidence": 0.90, "bbox": (50, 60, 70, 80)},
            {"label": "book", "confidence": 0.90, "bbox": (50, 60, 70, 80)},
            {"label": "scissors", "confidence": 0.90, "bbox": (50, 60, 70, 80)},
            {"label": "mouse", "confidence": 0.90, "bbox": (50, 60, 70, 80)},
        ]
        data.append((image_path, objects))

    return data

def test_db_insert_objects():
    n_images = 200
    test_data = generate_test_data(n_images)
    
    for image_path, objects in test_data:
        db.insert_detection(image_path, objects)

    total = MAX_OBJECT_ENTRIES * len(test_data[0][1])
    assert total == db.get_detected_object_count()


if __name__ == "__main__":
    test_db_insert_objects()