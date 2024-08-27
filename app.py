import streamlit as st
import torch
from config import Config
from model import VoiceRecognitionModel
from data_utils import preprocess_audio

@st.cache_resource
def load_model():
    model = VoiceRecognitionModel(Config.NUM_SPEAKERS)
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()
    return model

def run_app():
    model = load_model()
    st.title("Voice Recognition App")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        audio_tensor = preprocess_audio(uploaded_file)
        
        with torch.no_grad():
            output = model(audio_tensor)
            predicted_speaker = torch.argmax(output, dim=1).item()

        st.write(f"Predicted Speaker: {predicted_speaker}")
        if predicted_speaker in range(Config.NUM_SPEAKERS):
            st.success("Access granted!")
        else:
            st.error("Access denied. Voice not recognized.")