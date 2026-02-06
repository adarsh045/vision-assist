import cv2
import threading
import keyboard
import re
from playsound3 import playsound
from visionassist.model.yolo import YOLOModel
from visionassist.stt.recorder import AudioRecorder
from visionassist.stt.model import SpeechToTextModel
from visionassist.llm.model import getModel
from visionassist.tts.model import TextToSpeechModel
from visionassist.config import ENVITRONMENT, WHISPER_ACCESS_MODE, WHISPER_MODEL_NAME, ALLOWED_LABELS, LLM_MODEL_PROVIDER
from visionassist.logger import logger
from visionassist.memory.database import Database

def clean_text(text: str) -> str:
    text = text.lower()
    return re.sub(r'[^A-Za-z\s]', '', text)

class VisionAssistApp:
    """Main application class integrating YOLO detection and audio recording."""
    
    def __init__(self):
        self.yolo_model = YOLOModel()
        self.db = Database()
        self.audio_out = TextToSpeechModel()
        self.llm_model = getModel(LLM_MODEL_PROVIDER)
      
        # Initialize audio recorder
        try:
            self.recorder = AudioRecorder(output_file="user_audio.wav")
            logger.info("AudioRecorder initialized successfully")
        except ValueError as e:
            logger.error(f"Failed to initialize AudioRecorder: {e}")
            self.recorder = None
        
        # Initialize speech-to-text model
        if WHISPER_ACCESS_MODE == "online":
            self.stt_model = SpeechToTextModel(
                url=ENVITRONMENT["WHISPER_API_URL"],
                api_key=ENVITRONMENT["WHISPER_API_KEY"],
                mode="online"
            )
        else:
            self.stt_model = SpeechToTextModel(
                model_size=WHISPER_MODEL_NAME,
                mode="offline"
            )
        
        self.is_recording = False
        self.running = True
    
    def handle_recording(self):
        """Handle audio recording toggle."""
        if self.recorder is None:
            logger.warning("Recorder not available")
            return
        
        if not self.is_recording:
            # Start recording
            self.recorder.start_recording()
            self.is_recording = True
        else:
            # Stop recording and transcribe
            self.is_recording = False
            logger.info("Getting audio bytes and transcribing...")
            
            audio_bytes = self.recorder.get_audio_bytes_in_memory()
            
            if audio_bytes:
                # Transcribe in a separate thread to avoid blocking
                threading.Thread(
                    target=self.transcribe_audio, 
                    args=(audio_bytes,), 
                    daemon=True
                ).start()
    
    def transcribe_audio(self, audio_bytes):
        """Transcribe audio bytes and log the result."""
        try:
            # transcribe_from_bytes works for both online and offline modes
            result = self.stt_model.transcribe_from_bytes(audio_bytes)

            if result:
                # Handle different response formats
                if 'transcription' in result and isinstance(result['transcription'], dict):
                    transcription = result['transcription'].get('text', '')
                    
                    label = self.llm_model.predict_label(transcription)

                    if label in ALLOWED_LABELS:
                    
                        object_data = self.db.get_latest_objects(object_name=label, limit=1)
                        # logger.info(f"Object: {object_data[0].object_name}, Confidence: {object_data[0].confidence}, BBox: {object_data[0].bbox}")
                        # logger.info(f"Image path: {object_data[0].detection.image_path}")
                        try:
                            analysis = self.llm_model.analyze_image(object_data[0].detection.image_path, label)

                            logger.info(f"Analysis: {analysis}")
                            audio_path = self.audio_out.generate_audio(analysis)
                            playsound(audio_path)
                        except Exception as e:
                            logger.error(f"Error during image analysis or audio generation: {e}")
            else:
                logger.warning("No transcription result received")
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
    
    def keyboard_listener(self):
        """Listen for keyboard events."""
        logger.info("Keyboard listener started")
        logger.info("Press '2' to START/STOP recording")
        
        while self.running:
            try:
                keyboard.wait('2')
                self.handle_recording()
            except Exception as e:
                logger.error(f"Keyboard listener error: {e}")
                break

    def input_listener(self):
        """Listen for keyboard events. For giving input prompts instead of recording."""
        logger.info("Input listener started")
        logger.info("Press '1' to START/STOP input mode")
        
        while self.running:
            try:
                keyboard.wait('1')
                self.handle_input()
            except Exception as e:
                logger.error(f"Input listener error: {e}")
                break

    def handle_input(self):
        try:
            query = input("Enter your query : ")

            label = self.llm_model.predict_label(query)

            if label in ALLOWED_LABELS:
            
                object_data = self.db.get_latest_objects(object_name=label, limit=1)
                # logger.info(f"Object: {object_data[0].object_name}, Confidence: {object_data[0].confidence}, BBox: {object_data[0].bbox}")
                # logger.info(f"Image path: {object_data[0].detection.image_path}")
                try:
                    analysis = self.llm_model.analyze_image(object_data[0].detection.image_path, label)

                    logger.info(f"Analysis: {analysis}")
                    
                    # audio_path = self.audio_out.generate_audio(analysis)
                    
                    # playsound(audio_path)
                except Exception as e:
                    logger.error(f"Error during image analysis or audio generation: {e}")
        except Exception as e:
            logger.error(f"Input handling error: {e}")

    def realtime_stream(self):
        """Run the real-time video detection stream."""
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            logger.error("Error: Could not open webcam.")
            return

        # Start keyboard listener in separate thread
        if self.recorder:
            keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
            keyboard_thread.start()

            keyboard_thread_input = threading.Thread(target=self.input_listener, daemon=True)
            keyboard_thread_input.start()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Error: Could not read frame from webcam.")
                    break

                detections = self.yolo_model.detect(frame)
                
                img_path = self.yolo_model.save_with_bbox(frame, detections)

                if detections and img_path:
                    self.db.insert_detection(image_path=img_path, objects=detections)
                
                # Add recording indicator to frame
                if self.is_recording:
                    cv2.putText(frame, "REC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 0, 255), 2)
                    cv2.circle(frame, (70, 20), 8, (0, 0, 255), -1)
                
                cv2.imshow('YOLO Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.running = False
            if self.is_recording and self.recorder:
                self.recorder.stop_recording()
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Application closed")

def realtime_stream():
    """Legacy function for backward compatibility."""
    app = VisionAssistApp()
    app.realtime_stream()


if __name__ == "__main__":
    realtime_stream()