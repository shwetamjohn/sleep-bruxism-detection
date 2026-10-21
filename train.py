"""
Training Script for Sleep Bruxism Detection
Trains LSKNet (or ResNet baseline) on CWT Scalogram images with PyTorch and GPU acceleration.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.models import build_model

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")
        
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    val_loader = None
    if os.path.isdir(val_dir):
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
    return train_loader, val_loader

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=["Normal", "Bruxism"], zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return val_loss, f1, report, cm

def main():
    parser = argparse.ArgumentParser(description="Train LSKNet on CWT Scalograms for Bruxism Detection")
    parser.add_argument("--data_dir", type=str, default="data/dataset", help="Path to dataset root folder containing train/ and val/")
    parser.add_argument("--model_name", type=str, default="lsknet_t", help="Model architecture: 'lsknet_t', 'lsknet_s', 'resnet18'")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pth", help="Path to save best weights")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Prepare Data Loaders
    print(f"Loading data from: {args.data_dir}")
    train_loader, val_loader = get_data_loaders(args.data_dir, batch_size=args.batch_size)
    print(f"Classes found: {train_loader.dataset.classes}")
    
    # 2. Build Model
    print(f"Building model: {args.model_name}")
    model = build_model(model_name=args.model_name, num_classes=2, pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_f1 = 0.0
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
        if val_loader:
            val_loss, val_f1, report, cm = evaluate(model, val_loader, criterion, device)
            print(f"              Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), args.save_path)
                print(f"              [Saved new best model checkpoint to {args.save_path}]")
                
    print("\nTraining completed.")
    if val_loader:
        print("\nFinal Validation Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)

if __name__ == "__main__":
    main()
