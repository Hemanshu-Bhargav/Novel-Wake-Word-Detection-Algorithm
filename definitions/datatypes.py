from dataclasses import dataclass
import numpy as np

@dataclass
class AudioChunk:
    """Standardized payload for passing audio between modules."""
    data: np.ndarray
    sample_rate: int
    is_speech: bool = False