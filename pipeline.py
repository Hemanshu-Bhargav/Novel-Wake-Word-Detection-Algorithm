import numpy as np
from definitions.datatypes import AudioChunk
from definitions.interfaces import ZCRInterface, VADInterface, KeywordSpotterInterface

class AudioPipeline:
    def __init__(self, zcr: ZCRInterface, vad: VADInterface, kws: KeywordSpotterInterface):
        self.zcr = zcr
        self.vad = vad
        self.kws = kws
        self.audio_buffer = []

    def process_chunk(self, chunk: AudioChunk):
        """Processes a single incoming audio chunk from the live stream."""
        
        # 1. Fast Filter: Zero Crossing Rate
        if not self.zcr.check_activity(chunk):
            self._check_and_flush_buffer()
            return
            
        # 2. Moderate Filter: Deep VAD
        if self.vad.is_speech(chunk):
            self.audio_buffer.append(chunk.data)
        else:
            self._check_and_flush_buffer()

    def _check_and_flush_buffer(self):
        """If silence is detected after a period of speech, process the buffer."""
        if len(self.audio_buffer) > 0:
            self._trigger_keyword_spotting()

    def _trigger_keyword_spotting(self):
        # Concatenate buffered chunks into one continuous array
        full_audio = np.concatenate(self.audio_buffer)
        
        # 3. Heavy Filter: TDNN Keyword Spotting
        command = self.kws.predict(full_audio, sample_rate=16000)
        print(f"\n[TDNN] Detected Command: {command}")
        
        # Clear buffer for the next spoken word
        self.audio_buffer = []