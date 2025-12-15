import time
from visionassist.stt.recorder import AudioRecorder
from visionassist.stt.model import SpeechToTextModel
from visionassist.config import ENVITRONMENT
from visionassist.logger import logger

def test_recorder_with_whisper_bytes():
    try:
        recorder = AudioRecorder(output_file="test_recording.wav")
    except ValueError as e:
        logger.error(f"Failed to initialize recorder: {e}")
        return
    
    # Initialize Whisper in online mode
    stt_model = SpeechToTextModel(
        url=ENVITRONMENT["WHISPER_API_URL"],
        api_key=ENVITRONMENT["WHISPER_API_KEY"],
        mode="online"
    )
    
    # Start recording
    logger.info("Recording will start in 3 seconds...")
    time.sleep(3)
    
    recorder.start_recording()
    
    # Record for 10 seconds
    logger.info("Recording... (10 seconds)")
    time.sleep(10)
    
    audio_bytes = recorder.get_audio_bytes_in_memory()
    
    if audio_bytes is None:
        logger.error("Failed to get audio bytes")
        return
    
    result = stt_model.transcribe_from_bytes(audio_bytes)
    
    logger.info(f"Transcription Result: {result}")

    assert "transcription" in result

if __name__ == "__main__":
    test_recorder_with_whisper_bytes()