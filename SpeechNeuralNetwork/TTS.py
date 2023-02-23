import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
from rich.progress import track
import os
import sys


class WordProcessing:
    def __init__(self) -> None:
        pass

    def FeedForward(self,Input):
        pass

    def BackwardPropagation(self):
        pass

    def load(self):
        pass

    def save(self):
        pass


class EncoderNetwork(WordProcessing):
    def __init__(self) -> None:
        super().__init__()
    
    def FeedForward(self,Input):
        pass

    def BackwardPropagation(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

class SpeechSynthesisNetwork(EncoderNetwork):
    def __init__(self) -> None:
        super().__init__()
    
    def FeedForward(self,Input):
        pass

    def BackwardPropagation(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

class VocoderNetwork(SpeechSynthesisNetwork):
    def __init__(self) -> None:
        super().__init__()
    
    def FeedForward(self,Input):
        pass

    def BackwardPropagation(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

if __name__ == "__main__":
    word_processing = WordProcessing()
    encoder = EncoderNetwork()
    synthesis_network = SpeechSynthesisNetwork()
    vocoder = VocoderNetwork()
