import torch
import torch.nn as nn
from config import Config
from model import VoiceRecognitionModel
from data_utils import prepare_data_loaders

def train_model(file_paths, labels):
    train_loader, val_loader = prepare_data_loaders(file_paths, labels)
    
    model = VoiceRecognitionModel(Config.NUM_SPEAKERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Validation Accuracy: {100.*correct/total:.2f}%")

    torch.save(model.state_dict(), Config.MODEL_PATH)
    return model