import torch
import torch.nn as nn
import torchvision.models as models

def create_vgg16_model(num_classes):
    """
    Creates and returns a VGG16 model with custom classification head
    """
    # Load pretrained VGG16
    model = models.vgg16(pretrained=True)
    
    # Freeze feature layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Modify classifier
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, num_classes)
    )
    
    return model