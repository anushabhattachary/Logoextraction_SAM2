# checkpoints/load_model.py

import torch

def load_model(model, optimizer=None, path="checkpoints/best_model.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    accuracy = checkpoint.get('accuracy', 0.0)
    print(f"📦 Loaded model from {path} (Epoch {epoch}, Accuracy {accuracy:.2f}%)")
    return model, optimizer, epoch, accuracy
