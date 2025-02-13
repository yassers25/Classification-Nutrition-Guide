import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriculturalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self._load_images()
    
    def _is_valid_image(self, path):
        """Vérifie si le fichier est une image valide."""
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning(f"Image invalide {path}: {str(e)}")
            return False
    
    def _load_images(self):
        """Charge les chemins des images valides."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if Path(img_path).suffix in valid_extensions and self._is_valid_image(img_path):
                    self.images.append((img_path, cls))
                
        logger.info(f"Chargé {len(self.images)} images valides de {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, cls = self.images[idx]
        
        try:
            # Conversion explicite en RGB pour gérer tous les formats
            with Image.open(img_path) as image:
                image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            label = self.class_to_idx[cls]
            return image, label
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {img_path}: {str(e)}")
            # Retourner une image alternative en cas d'erreur
            return self.__getitem__((idx + 1) % len(self))