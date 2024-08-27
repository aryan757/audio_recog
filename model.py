import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class VoiceRecognitionModel(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(768, num_speakers)

    def forward(self, x):
        features = self.wav2vec(x).last_hidden_state
        pooled_features = torch.mean(features, dim=1)
        return self.classifier(pooled_features)
