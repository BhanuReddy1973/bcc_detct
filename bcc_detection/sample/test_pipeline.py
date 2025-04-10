import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support, classification_report
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import torch.serialization

# Add safe globals for numpy scalar types
torch.serialization.add_safe_globals([np._core.multiarray.scalar])

# Import our sample data loader
from sample_data_loader import BCCDataset, create_data_loaders

class EnhancedBCCModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Enhanced convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Enhanced fully connected layers
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(-1, 512 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in tqdm(train_loader, desc="Training"):
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)[:, 1].detach()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    
    return total_loss / len(train_loader), accuracy, precision, recall, f1, auc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Handle edge cases for precision, recall, and F1
    if len(np.unique(all_preds)) == 1:
        precision = 0.0 if all_preds[0] == 1 else 1.0
        recall = 0.0 if all_preds[0] == 1 else 1.0
        f1 = 0.0
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0)
    
    auc = roc_auc_score(all_labels, all_probs)
    
    return total_loss / len(val_loader), accuracy, precision, recall, f1, auc

def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=['Normal', 'BCC'])
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('test_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.close()
    
    return accuracy, precision, recall, f1, auc, report

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_dir = Path(__file__).parent / "data"
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, 
        batch_size=16,  # Reduced batch size
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    # Calculate class weights
    labels = []
    for batch in train_loader:
        labels.extend(batch['label'].numpy())
    labels = np.array(labels)
    class_weights = torch.FloatTensor([1.0 / np.sum(labels == i) for i in range(2)])
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    # Initialize model
    model = EnhancedBCCModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)  # Lower learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.3, min_lr=1e-6)
    
    # Training loop
    best_val_auc = 0
    patience = 5  # Early stopping patience
    no_improve = 0
    metrics = {
        'train': {'loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []},
        'val': {'loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
    }
    
    for epoch in range(20):
        print(f"\nEpoch {epoch+1}/20")
        
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate_epoch(
            model, val_loader, criterion, device)
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # Save metrics
        for metric, value in zip(['loss', 'acc', 'prec', 'rec', 'f1', 'auc'],
                               [train_loss, train_acc, train_prec, train_rec, train_f1, train_auc]):
            metrics['train'][metric].append(value)
        for metric, value in zip(['loss', 'acc', 'prec', 'rec', 'f1', 'auc'],
                               [val_loss, val_acc, val_prec, val_rec, val_f1, val_auc]):
            metrics['val'][metric].append(value)
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")
        
        # Save best model and check early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': best_val_auc,
            }, 'best_model.pth')
            print(f"New best model saved with AUC: {best_val_auc:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    metrics_to_plot = ['loss', 'acc', 'prec', 'rec', 'f1', 'auc']
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        plt.plot(metrics['train'][metric], label='Train')
        plt.plot(metrics['val'][metric], label='Val')
        plt.title(f'{metric.capitalize()}')
        plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    # Load best model and test
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc, test_prec, test_rec, test_f1, test_auc, test_report = test_model(model, test_loader, device)
    print("\nTest Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print("\nClassification Report:")
    print(test_report)

if __name__ == "__main__":
    main() 