import whisper


def speech_to_text() -> str:
    """
    Records audio from the microphone
    and converts it into text using Whisper.
    """
    print("hii")

    import sounddevice as sd
    from scipy.io.wavfile import write

    print("Loading Whisper model...")

    model = whisper.load_model("base")

    sample_rate = 16000
    duration = 10

    print("\nSpeak your answer...")
    print(f"You have {duration} seconds.")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    print("Recording finished.")

    audio_file = "answer.wav"

    write(
        audio_file,
        sample_rate,
        audio
    )

    print("Transcribing...")

    result = model.transcribe(
        audio_file,
        language="en"
    )

    answer = result["text"].strip()

    return answer