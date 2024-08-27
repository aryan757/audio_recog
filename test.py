import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import time

# 1. Updated Preprocessing Function
def preprocess_voice(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfcc

# 2. Updated Neural Network Model
class VariableLengthVoiceRecognitionModel(nn.Module):
    def __init__(self):
        super(VariableLengthVoiceRecognitionModel, self).__init__()
        self.conv1 = nn.Conv1d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 3)  # Output layer for 3 classes

    def forward(self, x):
        x = x.transpose(1, 2)  # Change from (batch, time, features) to (batch, features, time)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# 3. Updated Training Function
def train_model():
    # Example: Process your three voice samples
    voice1_features = preprocess_voice(librosa.load('nirupam_audio.mp3')[0], sr=16000)
    voice2_features = preprocess_voice(librosa.load('aryan_audio.mp3')[0], sr=16000)
    voice3_features = preprocess_voice(librosa.load('ashutosh_audio.mp3')[0], sr=16000)

    # Prepare the training data
    train_data = [voice1_features, voice2_features, voice3_features]
    labels = torch.tensor([0, 1, 2])  # Assign unique labels to each voice

    model = VariableLengthVoiceRecognitionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):  # Train for 100 epochs
        optimizer.zero_grad()
        total_loss = 0
        for features, label in zip(train_data, labels):
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            output = model(features_tensor)
            loss = criterion(output, label.unsqueeze(0))
            total_loss += loss
        
        total_loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/100], Loss: {total_loss.item():.4f}')

    return model

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []

    def recv(self, frame):
        self.audio_buffer.append(frame.to_ndarray())
        return frame

def main():
    st.title("Voice Recognition System")

    model = train_model()  # Train the model when the app starts

    st.header("Voice Recording and Recognition")
    st.write("Click 'Start' to begin recording. The system will automatically stop after 5 seconds and process your voice.")

    webrtc_ctx = webrtc_streamer(
        key="voice_record",
        audio_processor_factory=AudioProcessor,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
        ),
    )

    if webrtc_ctx.audio_processor:
        if st.button("Start Recording"):
            st.session_state.recording = True
            st.warning("Recording for 5 seconds...")
            
            # Record for 5 seconds
            time.sleep(5)
            
            st.session_state.recording = False
            st.success("Recording complete. Processing...")

            # Process the recorded audio
            audio_data = np.concatenate(webrtc_ctx.audio_processor.audio_buffer)
            
            # Convert int16 audio data to float32
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Convert audio data to features
            try:
                features = preprocess_voice(audio_data, sr=16000)
                
                # Make prediction
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output = model(features_tensor)
                predicted_label = torch.argmax(output, dim=1).item()

                if predicted_label == 0:  # Adjust according to your labels
                    st.success("Voice recognized! Unlocked!")
                else:
                    st.error("Voice not recognized. Access Denied.")
            except Exception as e:
                st.error(f"An error occurred while processing the audio: {str(e)}")

            # Clear the audio buffer for the next recording
            webrtc_ctx.audio_processor.audio_buffer = []

if __name__ == "__main__":
    main()