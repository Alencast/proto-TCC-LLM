import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 5

print("Fale alguma coisa...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write("audio.wav", audio, SAMPLE_RATE)

print("Áudio gravado em audio.wav")