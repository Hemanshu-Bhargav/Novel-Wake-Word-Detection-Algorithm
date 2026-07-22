import pyaudio
import numpy as np
from pydub import AudioSegment

# Import definitions and engines
from definitions.datatypes import AudioChunk
from zero_crossing_rate.zcr_engine import BasicZCR
from vad.vad_engine import PyPIVADWrapper
from kws.tdnn_engine import SpeechBrainTDNN
from pipeline import AudioPipeline

def start_live_stream(pipeline: AudioPipeline):
    """Captures live audio, wraps it in PyDub, and feeds the pipeline."""
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK_SIZE = int(RATE * 0.1)  # 100ms chunks

    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print("Microphone active. Listening for keywords... (Press Ctrl+C to stop)")
    
    try:
        while True:
            # 1. Capture raw hardware bytes
            raw_bytes = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # 2. Wrap in PyDub's AudioSegment for manipulation
            pydub_segment = AudioSegment(
                data=raw_bytes,
                sample_width=audio_interface.get_sample_size(FORMAT),
                frame_rate=RATE,
                channels=CHANNELS
            )

            # (Optional) Apply PyDub manipulations here
            # e.g., pydub_segment = pydub_segment + 3  # Increase volume by 3dB

            # 3. Convert back to numpy array
            audio_np = np.frombuffer(pydub_segment.raw_data, dtype=np.int16)

            # 4. Construct payload and process
            chunk = AudioChunk(data=audio_np, sample_rate=RATE)
            pipeline.process_chunk(chunk)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        # Clean up audio interfaces
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

def main():
    # 1. Initialize concrete implementations
    zcr_module = BasicZCR(threshold=0.05)
    vad_module = PyPIVADWrapper()
    kws_module = SpeechBrainTDNN()

    # 2. Inject dependencies into the pipeline
    pipeline = AudioPipeline(zcr=zcr_module, vad=vad_module, kws=kws_module)
    
    # 3. Start hardware capture
    start_live_stream(pipeline)

if __name__ == "__main__":
    main()