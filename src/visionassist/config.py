# Configurations for the application
import os
from dotenv import load_dotenv
load_dotenv()

MAX_OBJECT_ENTRIES = 50

ALLOWED_LABELS = ['backpack', 'umbrella', 'handbag', 'suitcase', 'bottle', 'cup', 'laptop', 'mouse', 'cell phone', 'book', 'scissors']

MIN_CONFIDENCE = 0.6

IMAGE_DIR = "data/images/"

YOLO_MODEL_NAME = "yolov8s.pt"

WHISPER_MODEL_NAME = "small.en"

WHISPER_ACCESS_MODE = "online"  # options: 'offline', 'online'

ENVITRONMENT = {
    "type" : "production",  # options: 'development', 'production'
    "debug": False, # True or False
    "log_level": "INFO",  # options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    
    "WHISPER_API_URL": os.getenv("WHISPER_API_URL", None),
    "WHISPER_API_KEY": os.getenv("WHISPER_API_KEY", None)
}