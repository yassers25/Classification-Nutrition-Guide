# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import os
from datetime import datetime
from data_preparation import AgriculturalDataset
from config import Config
import logging
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import gc
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    # Configuration
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Chargement des données
    try:
        train_dataset = AgriculturalDataset(Config.TRAIN_DIR, transform=data_transforms['train'])
        val_dataset = AgriculturalDataset(Config.VAL_DIR, transform=data_transforms['val'])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"\nConfiguration de l'entraînement:")
        logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        logger.info(f"Batch size: {Config.BATCH_SIZE}")
        logger.info(f"Nombre d'époques: {Config.EPOCHS}")
        logger.info(f"Learning rate: {Config.LEARNING_RATE}")
        logger.info(f"Nombre de classes: {len(train_dataset.classes)}")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        return
    
    # Modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Modification pour la mémoire CPU
    if device.type == 'cpu':
        model = model.float()
    
    num_classes = len(train_dataset.classes)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.1
    )
    
    best_acc = 0.0
    start_time = time.time()
    
    # Boucle d'entraînement
    try:
        for epoch in range(Config.EPOCHS):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            # Barre de progression pour l'entraînement
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS} [Train]')
            for inputs, labels in train_pbar:
                try:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # Mise à jour de la barre de progression
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{torch.sum(preds == labels.data).item() / len(labels):.4f}'
                    })
                    
                except Exception as e:
                    logger.error(f"Erreur batch entraînement: {str(e)}")
                    continue
                
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS} [Val]')
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    try:
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        val_corrects += torch.sum(preds == labels.data)
                        
                        val_pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{torch.sum(preds == labels.data).item() / len(labels):.4f}'
                        })
                        
                    except Exception as e:
                        logger.error(f"Erreur batch validation: {str(e)}")
                        continue
            
            val_loss = val_loss / len(val_dataset)
            val_acc = val_corrects.double() / len(val_dataset)
            
            epoch_time = time.time() - epoch_start
            
            # Affichage des résultats
            logger.info(f'\nEpoch {epoch+1}/{Config.EPOCHS} - Temps: {epoch_time:.2f}s')
            logger.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            
            # Mise à jour du learning rate
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Learning rate actuel: {current_lr}')
            
            # Sauvegarde du meilleur modèle
            if val_acc > best_acc:
                best_acc = val_acc
                model_path = os.path.join(
                    Config.MODEL_SAVE_DIR,
                    f'model_acc_{val_acc:.3f}_epoch_{epoch+1}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'classes': train_dataset.classes
                }, model_path)
                logger.info(f'Meilleur modèle sauvegardé: {model_path}')
            
            # Nettoyage de la mémoire
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {str(e)}")
    
    total_time = time.time() - start_time
    logger.info(f'\nEntraînement terminé en {total_time/60:.2f} minutes')
    logger.info(f'Meilleure précision: {best_acc:.4f}')
    
    return model

if __name__ == '__main__':
    model = train_model()