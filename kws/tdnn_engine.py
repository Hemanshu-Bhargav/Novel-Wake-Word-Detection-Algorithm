"""
tdnn_engine.py

Implements the TDNN Keyword Spotting module using SpeechBrain.
"""

import torch
import numpy as np
from speechbrain.pretrained import EncoderClassifier

import sys
import os
from definitions.interfaces import TDNNInterface

class SpeechBrainTDNN(TDNNInterface):
    def __init__(self):
        print("Loading TDNN model from Hugging Face...")
        # Loads a pre-trained TDNN model trained on Google Speech Commands
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/google_speech_command_xvector"
        )

    def predict_command(self, audio_input: np.ndarray) -> str:
        """
        Adapter method to satisfy the TDNNInterface contract used by the pipeline.
        Defaults to 16kHz which is standard for Google Speech Commands.
        """
        return self.predict(audio_input, sample_rate=16000)

    def predict(self, audio_data: np.ndarray, sample_rate: int) -> str:
        # Convert numpy array to torch tensor
        signal = torch.tensor(audio_data).float()
        
        # SpeechBrain expects a batch dimension (batch_size, time)
        signal = signal.unsqueeze(0) 
        
        # Run inference
        out_prob, score, index, text_lab = self.classifier.classify_batch(signal)
        
        # Return the predicted command (e.g., "stop", "go", "left")
        return text_lab[0]