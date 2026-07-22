from typing import Protocol
import numpy as np
from .datatypes import AudioChunk

class ZCRInterface(Protocol):
    def check_activity(self, chunk: AudioChunk) -> bool:
        """Returns True if the zero-crossing rate suggests potential speech."""
        ...

class VADInterface(Protocol):
    def is_speech(self, chunk: AudioChunk) -> bool:
        """Runs a dedicated VAD model to confirm speech."""
        ...

class KeywordSpotterInterface(Protocol):
    def predict(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Passes the aggregated audio buffer to the TDNN and returns the word."""
        ...