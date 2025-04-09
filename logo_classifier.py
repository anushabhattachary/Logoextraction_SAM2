# models/logo_classifier.py

import torch.nn as nn
from torchvision import models

def get_logo_classifier(num_classes=10, pretrained=True):
    """
    Returns a MobileNetV2 model for logo classification.
    """
    model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
