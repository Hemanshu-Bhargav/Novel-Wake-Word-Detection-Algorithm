import torch
import numpy as np
from speechbrain.pretrained import EncoderClassifier

class SpeechBrainTDNN:
    def __init__(self):
        print("Loading TDNN model from Hugging Face...")
        # Loads a pre-trained TDNN model trained on Google Speech Commands
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/google_speech_command_xvector"
        )

    def predict(self, audio_data: np.ndarray, sample_rate: int) -> str:
        # Convert numpy array to torch tensor
        signal = torch.tensor(audio_data).float()
        
        # SpeechBrain expects a batch dimension (batch_size, time)
        signal = signal.unsqueeze(0) 
        
        # Run inference
        out_prob, score, index, text_lab = self.classifier.classify_batch(signal)
        
        # Return the predicted command (e.g., "stop", "go", "left")
        return text_lab[0]