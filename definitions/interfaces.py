"""
interfaces.py

This module defines the Abstract Base Classes (ABCs) for the Audio Processing Pipeline.
It enforces a strict contract for the components mapped out in the pipeline flowchart:
Raw Audio Stream -> Orchestrator -> ZCR -> VAD -> TDNN -> Output.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union

class ZCRInterface(ABC):
    """
    Filter: Time-domain zero crossing rate analysis.
    Used as an early-stage gate to drop frames that fall below a basic noise floor.
    """
    
    @abstractmethod
    def analyze(self, audio_buffer: np.ndarray) -> bool:
        """
        Analyzes the audio buffer's zero-crossing rate.
        
        Args:
            audio_buffer (np.ndarray): The raw audio signal array.
            
        Returns:
            bool: True if the ZCR meets the threshold (indicating potential speech), False otherwise.
        """
        pass


class VADInterface(ABC):
    """
    Classification: Neural/Statistical voice activity detection.
    Isolates human vocal segments from general noise.
    """
    
    @abstractmethod
    def is_speech(self, audio_buffer: np.ndarray) -> bool:
        """
        Detects if the provided audio frame contains human speech.
        
        Args:
            audio_buffer (np.ndarray): The audio signal array.
            
        Returns:
            bool: True if speech is detected, False otherwise.
        """
        pass


class TDNNInterface(ABC):
    """
    Inference: Time Delay Neural Network (TDNN) for keyword spotting/classification.
    Executes the final classification on valid speech segments.
    """
    
    @abstractmethod
    def predict_command(self, audio_input: Any) -> str:
        """
        Classifies the speech segment into a specific command.
        
        Args:
            audio_input (Any): The pre-processed audio segment (e.g., tensor, file path, or numpy array).
            
        Returns:
            str: The predicted speech command label.
        """
        pass


class PipelineInterface(ABC):
    """
    Control: Manages buffering, sequencing, and conditional execution of the pipeline modules.
    """
    
    @abstractmethod
    def process_stream(self, audio_stream: Any):
        """
        Reads from the continuous PCM audio buffer input and orchestrates the filtering,
        VAD, and TDNN classification sequentially.
        
        Args:
            audio_stream (Any): The incoming raw audio stream source.
        """
        pass