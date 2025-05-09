# Real-Time Speech-to-Text for Contact Center Automation

This project is a real-time speech-to-text transcription system using [OpenAI Whisper](https://github.com/openai/whisper). It's designed for customer support automation in contact centers, enabling real-time transcription from microphone input, even in noisy environments.

## Features

- Real-time microphone audio capture
- Robust speech-to-text transcription using Whisper
- Handles noisy environments effectively
- Easily extendable (CRM integration, sentiment analysis, speaker diarization)

## Requirements

Install dependencies using pip:

```bash
pip install openai-whisper sounddevice numpy soundfile
```

You also need to have `ffmpeg` installed. On Ubuntu/Debian:

```bash
sudo apt install ffmpeg
```

On MacOS with Homebrew:

```bash
brew install ffmpeg
```

## Usage

Save the following code in `main.py` and run it:

```python
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import tempfile
import soundfile as sf

model = whisper.load_model("base")
q = queue.Queue()
samplerate = 16000
blocksize = 16000 * 5  # 5 seconds of audio

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=audio_callback):
        print("Listening...")
        while True:
            time.sleep(0.1)

def transcribe_audio():
    while True:
        if not q.empty():
            audio_chunk = q.get()
            audio_chunk = np.squeeze(audio_chunk)
            audio_int16 = (audio_chunk * 32767).astype(np.int16)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_int16, samplerate)
                result = model.transcribe(f.name)
                print("Transcription:", result["text"])

threading.Thread(target=record_audio, daemon=True).start()
transcribe_audio()
```

Run the app with:

```bash
python main.py
```

## Future Extensions

- Speaker diarization (agent vs. customer)
- Sentiment analysis and intent recognition
- CRM integration (Salesforce, Zendesk, etc.)

## License

MIT License
