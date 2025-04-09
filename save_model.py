# checkpoints/save_model.py

import torch
import os

def save_model(model, optimizer, epoch, accuracy, path="checkpoints/best_model.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy
    }
    torch.save(checkpoint, path)
    print(f"✅ Model saved at {path}")
