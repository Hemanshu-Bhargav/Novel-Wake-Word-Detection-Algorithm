"""
pipeline_modules.py

Provides concrete implementations for the audio processing pipeline interfaces.
Includes the Zero-Crossing Rate (ZCR) filter, Voice Activity Detection (VAD), 
and the Pipeline that wires the flow: Stream -> ZCR -> VAD -> TDNN.
"""

import numpy as np
import logging

try:
    import webrtcvad
except ImportError:
    logging.warning("webrtcvad not installed. Run: pip install webrtcvad. Using mock VAD for now.")
    webrtcvad = None

from interfaces import ZCRInterface, VADInterface, PipelineInterface, TDNNInterface

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class ZCRModule(ZCRInterface):
    """
    Concrete implementation of the ZCR filter.
    Calculates the rate at which the signal changes sign.
    """
    def __init__(self, min_threshold: float = 0.02, max_threshold: float = 0.4):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def analyze(self, audio_buffer: np.ndarray) -> bool:
        """
        Returns True if the zero-crossing rate falls within a range typical for speech.
        Too low = silence/hum, Too high = white noise/hiss.
        """
        # Calculate the zero-crossing rate of the numpy array
        crossings = np.nonzero(np.diff(np.sign(audio_buffer)))[0]
        zcr = len(crossings) / len(audio_buffer)
        
        is_valid = self.min_threshold <= zcr <= self.max_threshold
        logging.debug(f"ZCR Analysis: {zcr:.4f} -> {'Pass' if is_valid else 'Drop'}")
        return is_valid


class PyPIVADModule(VADInterface):
    """
    Concrete implementation of VAD using Google's webrtcvad PyPI package.
    """
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2):
        self.sample_rate = sample_rate
        if webrtcvad:
            self.vad = webrtcvad.Vad(aggressiveness)
        else:
            self.vad = None

    def is_speech(self, audio_buffer: np.ndarray) -> bool:
        """
        Converts the normalized float32 audio buffer to 16-bit PCM and checks for speech.
        """
        if not self.vad:
            # Fallback mock logic if webrtcvad is missing: simple energy threshold
            energy = np.sum(audio_buffer ** 2) / len(audio_buffer)
            return energy > 0.005

        # webrtcvad expects 16-bit mono PCM audio
        # Scale float32 [-1.0, 1.0] to int16 [-32768, 32767]
        pcm_data = (audio_buffer * 32767).astype(np.int16).tobytes()
        
        try:
            # webrtcvad requires frames to be exactly 10, 20, or 30 ms long.
            # Assuming the pipeline guarantees these chunk sizes.
            return self.vad.is_speech(pcm_data, self.sample_rate)
        except Exception as e:
            logging.error(f"VAD Error: {e}")
            return False


class Pipeline(PipelineInterface):
    """
    Manages the data flow between modules.
    Reads an audio stream, chunks it, and passes it through the conditional logic gates.
    """
    def __init__(self, zcr: ZCRInterface, vad: VADInterface, tdnn: TDNNInterface, chunk_size: int = 480):
        self.zcr = zcr
        self.vad = vad
        self.tdnn = tdnn
        self.chunk_size = chunk_size  # 480 samples @ 16kHz = 30ms frame (ideal for WebRTC VAD)
        self.speech_buffer = []

    def process_stream(self, audio_stream):
        """
        Consumes an iterable audio stream yielding raw samples.
        """
        logging.info("Starting pipeline processing...")
        
        # Buffer incoming samples into fixed-size frames
        current_frame = []
        for sample in audio_stream:
            current_frame.append(sample)
            
            # Once we have a full frame, process it
            if len(current_frame) == self.chunk_size:
                audio_array = np.array(current_frame, dtype=np.float32)
                current_frame = [] # Reset frame
                
                # Gate 1: ZCR (Threshold Met?)
                if not self.zcr.analyze(audio_array):
                    continue
                
                # Gate 2: VAD (Speech Detected?)
                if not self.vad.is_speech(audio_array):
                    continue
                
                # Accumulate valid speech frames
                self.speech_buffer.extend(audio_array)
                
                # If we've accumulated enough speech (e.g., 1 second = 16000 samples)
                if len(self.speech_buffer) >= 16000:
                    logging.info("Valid speech segment isolated. Sending to TDNN...")
                    
                    # Inference: TDNN Keyword Spotting
                    speech_segment = np.array(self.speech_buffer, dtype=np.float32)
                    prediction = self.tdnn.predict_command(speech_segment)
                    
                    logging.info(f"*** PREDICTED COMMAND: {prediction} ***")
                    
                    # Clear buffer after processing
                    self.speech_buffer = []


if __name__ == "__main__":
    # A quick mock runner to demonstrate how to assemble the pipeline
    logging.info("Assembling pipeline components...")
    
    zcr_module = ZCRModule()
    vad_module = PyPIVADModule()
    
    # Mocking the TDNN just for the structural test
    class MockTDNN(TDNNInterface):
        def predict_command(self, audio_input):
            return "WAKE_WORD_DETECTED"
            
    tdnn_module = MockTDNN()
    
    pipeline = Pipeline(zcr_module, vad_module, tdnn_module)
    
    # Generate 1.5 seconds of mock audio (random noise + a sine wave representing a voice)
    logging.info("Generating mock audio stream (1.5 seconds)...")
    noise = np.random.normal(0, 0.01, 16000 * 2)
    voice = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 16000 * 2)) * 0.5
    mock_audio_stream = (noise + voice).tolist()
    
    pipeline.process_stream(mock_audio_stream)