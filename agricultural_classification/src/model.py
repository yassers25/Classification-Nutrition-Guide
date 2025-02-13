import torch
import torch.nn as nn
from torchvision import models
from config import Config

def create_model(num_classes):
    # Charger le modèle pré-entraîné
    model = models.__dict__[Config.MODEL_NAME](pretrained=True)
    
    # Geler les couches
    for param in model.parameters():
        param.requires_grad = False
    
    # Remplacer la dernière couche
    if Config.MODEL_NAME.startswith('resnet'):
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    return model
