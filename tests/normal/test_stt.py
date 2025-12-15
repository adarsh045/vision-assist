from visionassist.stt.model import SpeechToTextModel
from visionassist.config import ENVITRONMENT, WHISPER_MODEL_NAME

def test_stt_model_offline():
    stt_model = SpeechToTextModel(
        model_size=WHISPER_MODEL_NAME,
        mode="offline"
    )
    data = stt_model.transcribe_from_file("tests/assets/audio.mp3")
    
    assert "transcription" in data

def test_stt_model_online():
    stt_model = SpeechToTextModel(
        url=ENVITRONMENT["WHISPER_API_URL"],
        api_key=ENVITRONMENT["WHISPER_API_KEY"],
        mode="online"
    )
    data = stt_model.transcribe_from_api("tests/assets/audio.mp3")

    assert "transcription" in data

if __name__ == "__main__":
    test_stt_model_offline()
    test_stt_model_online()