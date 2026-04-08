"""
Deep Learning Models for Crop Yield Prediction
Implements DeepAgYieldNet, ShuffleNetV2, EfficientCapsNet, LAD-Net, and RegNet
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class DeepAgYieldNet(nn.Module):
    """
    Deep Agricultural Yield Network - Specialized for crop yield prediction
    Multi-layer neural network with agricultural domain-specific features
    """
    def __init__(self, input_size, num_classes, hidden_sizes=[256, 128, 64, 32]):
        super(DeepAgYieldNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ShuffleNetV2Block(nn.Module):
    """ShuffleNet V2 basic block adapted for tabular data"""
    def __init__(self, in_channels, out_channels):
        super(ShuffleNetV2Block, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(in_channels // 2, out_channels // 2),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Linear(in_channels // 2, out_channels // 2),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU()
        )
        
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        return torch.cat([out1, out2], dim=1)

class ShuffleNetV2(nn.Module):
    """
    ShuffleNet V2 adapted for crop prediction
    Efficient architecture with channel shuffle operations
    """
    def __init__(self, input_size, num_classes):
        super(ShuffleNetV2, self).__init__()
        
        # Initial layer to make input size even for shuffling
        if input_size % 2 != 0:
            self.input_adapter = nn.Linear(input_size, input_size + 1)
            input_size = input_size + 1
        else:
            self.input_adapter = nn.Identity()
        
        self.initial = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.shuffle_blocks = nn.Sequential(
            ShuffleNetV2Block(128, 128),
            ShuffleNetV2Block(128, 64),
            ShuffleNetV2Block(64, 32)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        x = self.input_adapter(x)
        x = self.initial(x)
        x = self.shuffle_blocks(x)
        x = self.classifier(x)
        return x

class EfficientCapsNet(nn.Module):
    """
    Efficient Capsule Network for crop prediction
    Uses capsule layers for better feature representation
    """
    def __init__(self, input_size, num_classes, num_capsules=8, capsule_dim=16):
        super(EfficientCapsNet, self).__init__()
        
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        # Primary capsules
        self.primary_caps = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_capsules * capsule_dim)
        )
        
        # Digital capsules
        self.digit_caps = nn.Linear(num_capsules * capsule_dim, num_classes * capsule_dim)
        
        # Reconstruction network
        self.reconstruction = nn.Sequential(
            nn.Linear(num_classes * capsule_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_size)
        )
        
    def squash(self, tensor):
        """Squashing function for capsules"""
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)
        
    def forward(self, x):
        # Primary capsules
        primary = self.primary_caps(x)
        primary = primary.view(-1, self.num_capsules, self.capsule_dim)
        primary = self.squash(primary)
        
        # Digital capsules
        primary_flat = primary.view(primary.size(0), -1)
        digit = self.digit_caps(primary_flat)
        digit = digit.view(-1, digit.size(1) // self.capsule_dim, self.capsule_dim)
        digit = self.squash(digit)
        
        # Classification output (length of capsules)
        output = torch.sqrt((digit ** 2).sum(dim=-1))
        
        return output

class LADNet(nn.Module):
    """
    Location-Aware Dense Network for crop prediction
    Incorporates spatial and temporal awareness for agricultural data
    """
    def __init__(self, input_size, num_classes):
        super(LADNet, self).__init__()
        
        # Location-aware feature extraction
        self.location_aware = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Dense blocks
        self.dense_block1 = self._make_dense_block(128, 4, 32)
        self.dense_block2 = self._make_dense_block(128 + 4*32, 4, 16)
        self.dense_block3 = self._make_dense_block(128 + 4*32 + 4*16, 4, 8)
        
        # Final classifier
        final_features = 128 + 4*32 + 4*16 + 4*8
        self.classifier = nn.Sequential(
            nn.Linear(final_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def _make_dense_block(self, in_features, num_layers, growth_rate):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.BatchNorm1d(in_features + i * growth_rate),
                nn.ReLU(),
                nn.Linear(in_features + i * growth_rate, growth_rate),
                nn.Dropout(0.2)
            ))
        return nn.ModuleList(layers)
        
    def _forward_dense_block(self, x, dense_block):
        features = [x]
        for layer in dense_block:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)
        
    def forward(self, x):
        x = self.location_aware(x)
        x = self._forward_dense_block(x, self.dense_block1)
        x = self._forward_dense_block(x, self.dense_block2)
        x = self._forward_dense_block(x, self.dense_block3)
        x = self.classifier(x)
        return x

class RegNet(nn.Module):
    """
    RegNet (Regular Network) for crop prediction
    Uses regular, parameterized network design
    """
    def __init__(self, input_size, num_classes, width_mult=1.0, depth_mult=1.0):
        super(RegNet, self).__init__()
        
        # Calculate widths based on RegNet design principles
        widths = [int(64 * width_mult), int(128 * width_mult), int(256 * width_mult), int(128 * width_mult)]
        depths = [int(2 * depth_mult), int(3 * depth_mult), int(4 * depth_mult), int(2 * depth_mult)]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Linear(input_size, widths[0]),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU()
        )
        
        # Stages
        self.stages = nn.ModuleList()
        in_width = widths[0]
        
        for stage_idx, (width, depth) in enumerate(zip(widths, depths)):
            stage = []
            for block_idx in range(depth):
                stage.append(nn.Sequential(
                    nn.Linear(in_width, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ))
                in_width = width
            self.stages.append(nn.Sequential(*stage))
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1) if len(widths) > 0 else nn.Identity(),
            nn.Flatten(),
            nn.Linear(widths[-1], num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            for block in stage:
                residual = x
                x = block(x)
                # Add residual connection if dimensions match
                if x.shape == residual.shape:
                    x = x + residual
        
        # For RegNet, we need to adapt the head for 1D data
        x = x.unsqueeze(-1)  # Add dimension for adaptive pooling
        x = self.head(x)
        return x

class DeepLearningTrainer:
    """Trainer class for all deep learning models"""
    
    def __init__(self, model_type='deepagyieldnet'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = device
        self.results = {}
        
    def prepare_data(self, X, y):
        """Prepare data for training"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        return (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    
    def create_model(self, input_size, num_classes):
        """Create model based on type"""
        if self.model_type == 'deepagyieldnet':
            model = DeepAgYieldNet(input_size, num_classes)
        elif self.model_type == 'shufflenetv2':
            model = ShuffleNetV2(input_size, num_classes)
        elif self.model_type == 'efficientcapsnet':
            model = EfficientCapsNet(input_size, num_classes)
        elif self.model_type == 'ladnet':
            model = LADNet(input_size, num_classes)
        elif self.model_type == 'regnet':
            model = RegNet(input_size, num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train_model(self, X, y, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the model"""
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Create model
        input_size = X.shape[1]
        num_classes = len(np.unique(y))
        self.model = self.create_model(input_size, num_classes)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)
        
        # Training loop
        best_accuracy = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if epoch % 10 == 0:
                accuracy = self.evaluate_model(test_loader)
                scheduler.step(accuracy)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")
        
        # Final evaluation
        final_accuracy = self.evaluate_model(test_loader)
        
        self.results = {
            'model_type': self.model_type,
            'accuracy': final_accuracy,
            'best_accuracy': best_accuracy,
            'num_classes': num_classes,
            'input_size': input_size
        }
        
        logger.info(f"{self.model_type} training completed. Final accuracy: {final_accuracy:.4f}")
        return self.results
    
    def evaluate_model(self, test_loader):
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return correct / total
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predicted.cpu().numpy())
        confidence_scores = probabilities.max(dim=1)[0].cpu().numpy()
        
        return predicted_labels, confidence_scores
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'results': self.results
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath, input_size, num_classes):
        """Load trained model"""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.model_type = model_data['model_type']
        self.model = self.create_model(input_size, num_classes)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.results = model_data['results']
        
        logger.info(f"Model loaded from {filepath}")

def train_all_models(X, y, save_dir='models/deep_learning'):
    """Train all deep learning models and save results"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_types = ['deepagyieldnet', 'shufflenetv2', 'efficientcapsnet', 'ladnet', 'regnet']
    results = {}
    
    for model_type in model_types:
        try:
            trainer = DeepLearningTrainer(model_type)
            result = trainer.train_model(X, y)
            results[model_type] = result
            
            # Save model
            model_path = os.path.join(save_dir, f'{model_type}_model.pth')
            trainer.save_model(model_path)
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
            results[model_type] = {'error': str(e), 'accuracy': 0.0}
    
    # Save comparison results
    comparison_path = os.path.join(save_dir, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("All models training completed")
    return results

if __name__ == "__main__":
    # Test the models with sample data
    logger.info("Testing deep learning models...")
    
    # Create sample data
    np.random.seed(42)
    X_sample = np.random.randn(1000, 7)  # 7 features
    y_sample = np.random.choice(['rice', 'wheat', 'cotton', 'maize'], 1000)
    
    # Train all models
    results = train_all_models(X_sample, y_sample)
    print("Training results:", results)