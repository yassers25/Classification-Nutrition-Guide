import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np

class SegmentationPipeline:
    def __init__(self):
        # Charger Mask R-CNN pré-entraîné
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def segment_image(self, image):
        """
        Segmente l'image pour détecter les objets
        Returns: Liste de segments (crops) de l'image originale
        """
        # Préparer l'image
        image_tensor = F.to_tensor(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Obtenir les prédictions
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Extraire les masques et les boîtes
        masks = predictions[0]['masks']
        boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']

        # Filtrer les prédictions avec un score > 0.5
        mask_thresh = 0.5
        high_scores_idxs = torch.where(scores > mask_thresh)[0]

        # Découper l'image selon les boîtes détectées
        crops = []
        for idx in high_scores_idxs:
            box = boxes[idx].cpu().numpy().astype(int)
            crop = image.crop((box[0], box[1], box[2], box[3]))
            crops.append(crop)

        return crops

class SegmentationAndClassificationPipeline:
    def __init__(self, classifier_pipeline):
        self.segmentation = SegmentationPipeline()
        self.classifier = classifier_pipeline

    def process_image(self, image):
        """
        Segmente l'image puis classifie chaque segment
        """
        # Segmenter l'image
        segments = self.segmentation.segment_image(image)
        
        # Classifier chaque segment
        results = []
        for segment in segments:
            predictions = self.classifier.predict_image(segment)
            results.append({
                'segment': segment,
                'predictions': predictions
            })
        
        return results