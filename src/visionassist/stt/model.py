import os
import time
import whisper
import requests
import io
from visionassist.logger import logger

class SpeechToTextModel:
    def __init__(self, url:str=None, api_key:str=None, model_size:str="base.en", mode:str="offline"):
        self.model = None
        self.mode = mode
        if mode not in ["offline", "online"]:
            raise ValueError("Mode must be either 'offline' or 'online'")
        if mode == "online":
            self.url = url
            self.api_key = api_key
            logger.info("Whisper initialized for online transcription.")
        else:
            self.model = whisper.load_model(model_size, )
            logger.info(f"Whisper model '{model_size}' loaded for offline transcription.")

    def transcribe_from_file(self, audio_path:str=None, audio_bytes:bytes=None):
        """
        Transcribe audio using offline model from file path or bytes.
        
        Args:
            audio_path: Path to audio file (optional if audio_bytes provided)
            audio_bytes: Audio data as bytes (optional if audio_path provided)
        """
        try:
            if not self.model:
                raise ValueError("Whisper model is not loaded for offline transcription.")
            
            # Create temp file if using bytes
            temp_file = None
            if audio_bytes:
                temp_file = "temp_offline_audio.wav"
                with open(temp_file, 'wb') as f:
                    f.write(audio_bytes)
                audio_path = temp_file
            elif audio_path:
                if not isinstance(audio_path, str) or not os.path.isfile(audio_path):
                    raise ValueError("audio_path must be a string representing the file path.")
            else:
                raise ValueError("Either audio_path or audio_bytes must be provided.")
            
            start_time = time.time()
            result = self.model.transcribe(audio_path)
            end_time = time.time()
            
            # Clean up temp file if created
            if temp_file:
                try:
                    os.remove(temp_file)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove temporary file: {cleanup_error}")
            
            transcription = result['text']
            processing_time = end_time - start_time
            
            return {
                "transcription": { "text" : transcription },
                "timing":{
                    "transcribe_ms": processing_time * 1000
                }
            }
        except Exception as e:
            logger.error(e)
            return None

    def transcribe_from_api(self, audio_path:str=None, audio_bytes:bytes=None):
        """
        Transcribe audio using online API from file path or bytes.
        
        Args:
            audio_path: Path to audio file (optional if audio_bytes provided)
            audio_bytes: Audio data as bytes (optional if audio_path provided)
        """
        try:
            if not hasattr(self, 'url') or not hasattr(self, 'api_key') or not self.url or not self.api_key:
                raise ValueError("API URL and API key must be set for online mode.")
            
            headers = { "Authorization": f"Bearer {self.api_key}" }
            
            if audio_bytes:
                # Use bytes directly with io.BytesIO
                audio_file = io.BytesIO(audio_bytes)
                response = requests.post(
                    self.url, 
                    headers=headers, 
                    files={"file": ("audio.wav", audio_file, "audio/wav")}
                )
            elif audio_path:
                # Use file path
                if not isinstance(audio_path, str) or not os.path.isfile(audio_path):
                    raise ValueError("audio_path must be a string representing the file path.")
                
                with open(audio_path, 'rb') as f:
                    response = requests.post(
                        self.url, 
                        headers=headers, 
                        files={"file": f}
                    )
            else:
                raise ValueError("Either audio_path or audio_bytes must be provided.")
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
            data = response.json()
            return data
            
        except Exception as e:
            logger.error(e)
            return None

    def transcribe_from_bytes(self, audio_bytes: bytes):
        """
        Transcribe audio from bytes for both online and offline modes.
        Uses in-memory processing for online mode, creates temp file only for offline.
        
        Args:
            audio_bytes: Audio data as bytes in WAV format
            filename: Filename hint for online API
            
        Returns:
            dict: Transcription result
        """
        try:
            if not audio_bytes or not isinstance(audio_bytes, bytes):
                raise ValueError("audio_bytes must be bytes data.")
            
            if self.mode == "online":
                result = self.transcribe_from_api(audio_bytes=audio_bytes)
            else:
                result = self.transcribe_from_file(audio_bytes=audio_bytes)
            
            return result
            
        except Exception as e:
            logger.error(e)
            return None