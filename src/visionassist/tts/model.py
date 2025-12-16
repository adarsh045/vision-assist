from gtts import gTTS
import os

class TextToSpeechModel:
    """Text-to-Speech model using Google TTS"""
    
    def __init__(self, lang="en"):
        self.lang = lang
    
    def generate_audio(self, text, output_path="output.mp3"):
        """
        Convert text to speech and save as audio file
        
        Args:
            text (str): The text to convert to speech
            output_path (str): Path where the audio file will be saved (default: "output.mp3")
            
        Returns:
            str: Absolute path to the saved audio file
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        tts = gTTS(text=text, lang=self.lang)
        tts.save(output_path)
        
        return os.path.abspath(output_path)