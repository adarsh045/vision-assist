import os
from visionassist.tts.model import TextToSpeechModel

def test_tts_model():
    stt_model = TextToSpeechModel(lang='en')
    abs_path = stt_model.generate_audio("this is a sample test" , "data/output.mp3")

    assert isinstance(abs_path, str) and os.path.isfile(abs_path)

if __name__ == "__main__":
    test_tts_model()