import torch
import torchaudio
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from pydub import AudioSegment
import io
import os

def load_audio(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.wav':
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.numpy()[0]
    elif file_extension == '.mp3':
        # Load MP3 using pydub
        audio = AudioSegment.from_mp3(file_path)
        samples = np.array(audio.get_array_of_samples())
        
        # Convert to float32 and normalize
        waveform = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        sample_rate = audio.frame_rate
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Resample to target sample rate
    if sample_rate != Config.SAMPLE_RATE:
        waveform = librosa.resample(waveform, sample_rate, Config.SAMPLE_RATE)
    
    return torch.tensor(waveform)

def load_and_preprocess_data(file_paths, labels):
    features = []
    for file_path in file_paths:
        waveform = load_audio(file_path)
        features.append(waveform)
    return features, torch.tensor(labels)

# The rest of the file remains the same
def prepare_data_loaders(file_paths, labels, batch_size=32):
    features, labels = load_and_preprocess_data(file_paths, labels)
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.stack(X_train), y_train)
    val_dataset = TensorDataset(torch.stack(X_val), y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def preprocess_audio(audio_file):
    waveform = load_audio(audio_file)
    return waveform.unsqueeze(0)