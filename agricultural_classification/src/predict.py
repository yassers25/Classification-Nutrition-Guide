import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import torchvision

from model import create_model
from efficientnet_model import create_efficientnet_model
from vgg16_model import create_vgg16_model
from config import Config
from efficientnet_config import EfficientNetConfig
from vgg16_config import VGG16Config

class TripleModelPredictionPipeline:
    def __init__(self, resnet_path, efficientnet_path, vgg16_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = Config.get_classes()
        
        # Load all models
        self.resnet_model = self._load_resnet(resnet_path)
        self.efficientnet_model = self._load_efficientnet(efficientnet_path)
        self.vgg16_model = self._load_vgg16(vgg16_path)
        
        # Get transforms for all models
        self.resnet_transform = self._get_resnet_transforms()
        self.efficientnet_transform = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT.transforms()
        self.vgg16_transform = self._get_vgg16_transforms()

    def _load_resnet(self, model_path):
        model = create_model(len(self.classes))
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model.eval()
        return model

    def _load_efficientnet(self, model_path):
        model = create_efficientnet_model(len(self.classes))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _load_vgg16(self, model_path):
        model = create_vgg16_model(len(self.classes))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _get_resnet_transforms(self):
        return transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _get_vgg16_transforms(self):
        return transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize((VGG16Config.IMG_SIZE, VGG16Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image):
        """
        Predicts the class of an image using all three models
        Args:
            image: PIL Image object
        Returns:
            dict: Predictions from all models with their probabilities
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get predictions from ResNet
            resnet_tensor = self.resnet_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                resnet_outputs = self.resnet_model(resnet_tensor)
                resnet_probs = torch.nn.functional.softmax(resnet_outputs[0], dim=0)
                resnet_pred = torch.argmax(resnet_probs)

            # Get predictions from EfficientNet
            efficientnet_tensor = self.efficientnet_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                efficientnet_outputs = self.efficientnet_model(efficientnet_tensor)
                efficientnet_probs = torch.nn.functional.softmax(efficientnet_outputs[0], dim=0)
                efficientnet_pred = torch.argmax(efficientnet_probs)

            # Get predictions from VGG16
            vgg16_tensor = self.vgg16_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                vgg16_outputs = self.vgg16_model(vgg16_tensor)
                vgg16_probs = torch.nn.functional.softmax(vgg16_outputs[0], dim=0)
                vgg16_pred = torch.argmax(vgg16_probs)

            return {
                'resnet': {
                    'class': self.classes[resnet_pred],
                    'probabilities': resnet_probs
                },
                'efficientnet': {
                    'class': self.classes[efficientnet_pred],
                    'probabilities': efficientnet_probs
                },
                'vgg16': {
                    'class': self.classes[vgg16_pred],
                    'probabilities': vgg16_probs
                }
            }

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise