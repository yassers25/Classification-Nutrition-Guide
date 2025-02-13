import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from tqdm import tqdm

from vgg16_model import create_vgg16_model
from vgg16_config import VGG16Config

def train_vgg16():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((VGG16Config.IMG_SIZE, VGG16Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder('train', transform=transform)
    val_dataset = datasets.ImageFolder('validation', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=VGG16Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VGG16Config.BATCH_SIZE)
    
    # Create model
    model = create_vgg16_model(len(VGG16Config.get_classes()))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=VGG16Config.LEARNING_RATE)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(VGG16Config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.3f}, Accuracy: {acc:.3f}%')
        
        # Save model if better accuracy
        if acc > best_acc:
            best_acc = acc
            model_path = os.path.join(VGG16Config.MODEL_SAVE_DIR, 
                                    f'vgg16_model_acc_{acc:.3f}_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    train_vgg16()