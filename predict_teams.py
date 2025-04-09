# scripts/predict_teams.py

import os
import torch
from torchvision import transforms
from PIL import Image
from logo_classifier import get_logo_classifier

def predict_logo(image_path, model, class_names, transform, device):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()
    return class_names[predicted], confidence
