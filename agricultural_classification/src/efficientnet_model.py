import torch
import torchvision
from torch import nn

def create_efficientnet_model(num_classes):
    """Create and configure EfficientNet model"""
    model = torchvision.models.efficientnet_v2_s(
        weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    )
    
    # Freeze the features
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Rebuild classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes)
    )
    
    return model
