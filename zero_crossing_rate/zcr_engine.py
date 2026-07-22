import numpy as np
from definitions.datatypes import AudioChunk

class BasicZCR:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def check_activity(self, chunk: AudioChunk) -> bool:
        """
        Calculates the zero-crossing rate. 
        Replace with your specific custom implementation if needed.
        """
        # Count how many times the signal crosses zero
        zero_crossings = np.nonzero(np.diff(chunk.data > 0))[0]
        zcr_rate = len(zero_crossings) / len(chunk.data)
        
        return zcr_rate > self.threshold