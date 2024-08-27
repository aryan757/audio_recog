# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
# import librosa
# import streamlit as st

# # 1. Data Preparation
# def preprocess_audio(file_path):
#     audio, sr = librosa.load(file_path, sr=16000)
#     return audio

# class VoiceDataset(Dataset):
#     def __init__(self, file_paths, labels):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         audio = preprocess_audio(self.file_paths[idx])
#         inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
#         label = torch.tensor(self.labels[idx])
#         return inputs.input_values.squeeze(), label

# # 2. Model Training
# def train_model(file_paths, labels):
#     dataset = VoiceDataset(file_paths, labels)
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#     model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=len(set(labels)))
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
#     model.train()

#     for epoch in range(3):  # Fine-tune for 3 epochs
#         for batch in dataloader:
#             input_values, labels = batch
#             outputs = model(input_values, labels=labels)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#         print(f"Epoch {epoch+1} - Loss: {loss.item()}")

#     model.save_pretrained("fine-tuned-wav2vec2")
#     return model

# # 3. Prediction
# def predict_voice(model, processor, file_path):
#     audio = preprocess_audio(file_path)
#     inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         logits = model(inputs.input_values).logits
#     predicted_label = torch.argmax(logits, dim=-1).item()
#     return predicted_label

# # 4. Streamlit Frontend
# def main():
#     st.title("Voice Recognition Unlock System")

#     # Assuming you've already trained the model
#     model = Wav2Vec2ForSequenceClassification.from_pretrained("fine-tuned-wav2vec2")
#     processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

#     uploaded_file = st.file_uploader("Upload a voice sample", type=["wav", "mp3"])

#     if uploaded_file is not None:
#         st.audio(uploaded_file, format="audio/wav")
#         predicted_label = predict_voice(model, processor, uploaded_file)
#         if predicted_label == 0:  # Adjust according to your labels
#             st.success("Unlocked!")
#         else:
#             st.error("Access Denied.")

# if __name__ == "__main__":
#     # 5. Training
#     # Paths to your voice samples and their labels
#     file_paths = ["aryan_audio.mp3", "ashutosh_audio.mp3", "nirupam_audio.mp3"]  # Replace with actual paths
#     labels = [0, 1, 2]  # Unique labels for each voice

#     # Train the model (comment this out after initial training to just use the model)
#     model = train_model(file_paths, labels)

#     # Start Streamlit app
#     main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import librosa
import streamlit as st

# 1. Data Preparation
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    return audio

class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio = preprocess_audio(self.file_paths[idx])
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        label = torch.tensor(self.labels[idx])
        return inputs.input_values.squeeze(0), label

def collate_fn(batch):
    input_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    input_values_padded = pad_sequence(input_values, batch_first=True)
    
    return input_values_padded, torch.tensor(labels)

# 2. Model Training
def train_model(file_paths, labels):
    dataset = VoiceDataset(file_paths, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=len(set(labels)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    for epoch in range(3):  # Fine-tune for 3 epochs
        for batch in dataloader:
            input_values, labels = batch
            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1} - Loss: {loss.item()}")

    model.save_pretrained("fine-tuned-wav2vec2")
    return model

# 3. Prediction
def predict_voice(model, processor, file_path):
    audio = preprocess_audio(file_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    return predicted_label

# 4. Streamlit Frontend
def main():
    st.title("Voice Recognition Unlock System")

    # Assuming you've already trained the model
    model = Wav2Vec2ForSequenceClassification.from_pretrained("fine-tuned-wav2vec2")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    uploaded_file = st.file_uploader("Upload a voice sample", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        predicted_label = predict_voice(model, processor, uploaded_file)
        if predicted_label == 0:  # Adjust according to your labels
            st.success("Unlocked!")
        else:
            st.error("Access Denied.")

if __name__ == "__main__":
    # 5. Training
    file_paths = ["aryan_audio.mp3", "ashutosh_audio.mp3", "nirupam_audio.mp3"]  # Replace with actual paths
    labels = [0, 1, 2]  # Unique labels for each voice

    # Train the model (comment this out after initial training to just use the model)
    model = train_model(file_paths, labels)

    # Start Streamlit app
    main()

