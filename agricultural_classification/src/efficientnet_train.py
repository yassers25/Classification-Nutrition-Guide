import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import time
from efficientnet_config import EfficientNetConfig
from efficientnet_model import create_efficientnet_model

def train_efficientnet():
    # Enable debug mode for more verbose output
    torch.autograd.set_detect_anomaly(True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get transforms
    transform = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT.transforms()

    # Load datasets with error handling
    try:
        train_dataset = ImageFolder(EfficientNetConfig.TRAIN_DIR, transform=transform)
        val_dataset = ImageFolder(EfficientNetConfig.VAL_DIR, transform=transform)
        print(f"Number of training images: {len(train_dataset)}")
        print(f"Number of validation images: {len(val_dataset)}")
        print(f"Number of classes: {len(train_dataset.classes)}")
        print(f"Classes: {train_dataset.classes}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Create dataloaders with fewer workers for debugging
    train_loader = DataLoader(
        train_dataset,
        batch_size=EfficientNetConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Reduced for debugging
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=EfficientNetConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=2,  # Reduced for debugging
        pin_memory=True
    )

    # Initialize model with error handling
    try:
        model = create_efficientnet_model(len(train_dataset.classes))
        model = model.to(device)
        print("Model created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=EfficientNetConfig.LEARNING_RATE)

    # Training loop
    best_acc = 0.0
    start_time = time.time()
    
    print("Starting training...")
    for epoch in range(EfficientNetConfig.EPOCHS):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        processed_batches = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            batch_start = time.time()
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                processed_batches += 1
                
                if i % 10 == 9:
                    avg_loss = running_loss / 10
                    batch_time = time.time() - batch_start
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, '
                          f'Loss: {avg_loss:.3f}, '
                          f'Batch Time: {batch_time:.2f}s')
                    running_loss = 0.0
                
            except Exception as e:
                print(f"Error in training batch {i}: {e}")
                continue

        # Validation after each epoch
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s')
        print(f'Validation accuracy: {accuracy:.2f}%')
        print(f'Average validation loss: {val_loss/len(val_loader):.3f}')

        # Save model if it's the best so far
        if accuracy > best_acc:
            best_acc = accuracy
            model_save_path = os.path.join(
                EfficientNetConfig.MODEL_SAVE_DIR,
                f'efficientnet_model_acc_{accuracy:.3f}_epoch_{epoch+1}.pth'
            )
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved model to {model_save_path}')

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    train_efficientnet()