import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ResNet18Trainer:
    def __init__(self, train_dir, val_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
        # Configure device and CUDA optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            # Set current device
            torch.cuda.set_device(0)
            # Clear GPU cache
            torch.cuda.empty_cache()
            
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Data paths
        self.train_dir = train_dir
        self.val_dir = val_dir
        
        # Initialize transformers
        self.train_transform, self.val_transform = self._build_transforms()
        
        # Load data
        self.train_loader, self.val_loader, self.num_classes = self._load_data()
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=3
        )
        
        # Record best accuracy
        self.best_acc = 0.0
        
    def _build_transforms(self):
        """Build data transformers"""
        # ImageNet normalization parameters
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        return train_transform, val_transform
    
    def _load_data(self):
        """Load and prepare datasets"""
        # Check data directories
        if not os.path.isdir(self.train_dir):
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
        if not os.path.isdir(self.val_dir):
            raise FileNotFoundError(f"Validation directory not found: {self.val_dir}")

        # Load datasets
        train_dataset = datasets.ImageFolder(
            root=self.train_dir,
            transform=self.train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=self.val_dir,
            transform=self.val_transform
        )
        
        # Create data loaders with pin_memory for faster data transfer to GPU
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,           # The order of all training images is shuffled
            num_workers=4,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logging.info(f"Number of classes: {len(train_dataset.classes)}")
        logging.info(f"Class names: {train_dataset.classes}")
        logging.info(f"Number of training samples: {len(train_dataset)}")
        logging.info(f"Number of validation samples: {len(val_dataset)}")

        # Validate image file existence
        for dataset in [train_dataset, val_dataset]:
            for path, _ in dataset.samples:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image file not found: {path}")
        
        return train_loader, val_loader, len(train_dataset.classes)
    
    def _build_model(self):
        """Build and initialize the model"""
        # Load pretrained ResNet18
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the final fully connected layer to match our number of classes
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, self.num_classes)
        )
        
        model = model.to(self.device)
        return model
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        # Start CUDA event timer
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        
        return epoch_loss, epoch_acc.item()
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        
        return epoch_loss, epoch_acc.item()
    
    def save_checkpoint(self, epoch, acc):
        """Save model checkpoint"""
        # Create checkpoint directory
        checkpoint_dir = 'resnet18_checkpoint'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': acc,
            'classes': self.train_loader.dataset.classes
        }
        
        # Save current checkpoint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f'resnet18_checkpoint_{timestamp}_acc_{acc:.4f}.pth'
        save_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(checkpoint, save_path)
        logging.info(f"Model saved: {save_path}")
        
        # Save best model
        if acc > self.best_acc:
            self.best_acc = acc
            best_name = 'best_resnet18_model.pth'
            best_path = os.path.join(checkpoint_dir, best_name)
            torch.save(checkpoint, best_path)
            logging.info(f"Best model updated: {best_path}")
    
    def train(self):
        """Train the model"""
        logging.info(f"Starting training - Using device: {self.device}")
        logging.info(f"Total training epochs: {self.num_epochs}")
        
        for epoch in range(1, self.num_epochs + 1):
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Log progress
            logging.info(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_acc)
        
        logging.info(f"Training completed! Best validation accuracy: {self.best_acc:.4f}")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # CUDA configuration
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        # torch.backends.cuda.max_memory_allocated()
        torch.backends.cudnn.benchmark = True       # Enable cudnn auto-tuner
        torch.backends.cudnn.deterministic = False  # Disable deterministic mode for improved performance
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logging.warning("CUDA is not available. Training will be performed on CPU.")
    
    # Training parameters
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, './dataset/Training')
    val_dir = os.path.join(base_dir, './dataset/Validation')
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001

    # Run the ResNet18 training
    logging.info("Starting ResNet18 training...")
    trainer = ResNet18Trainer(
        train_dir=train_dir,
        val_dir=val_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Start training
    trainer.train()
    logging.info("ResNet18 training completed.")

if __name__ == '__main__':
    main()
