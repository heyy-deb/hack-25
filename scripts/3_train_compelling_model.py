#!/usr/bin/env python3
"""
Step 3: Train Compelling Scene Classifier
==========================================

Trains a binary classifier to identify compelling movie scenes
based on the CondensedMovies dataset.

The model learns what makes a scene "movieclips-worthy" - the kind
of moment that makes viewers want to watch the full movie.

Usage:
    # Quick test with dummy data
    python 3_train_compelling_model.py --use-dummy --num-epochs 3
    
    # Train on real data
    python 3_train_compelling_model.py --data-dir data/training_data
    
    # Evaluate existing model
    python 3_train_compelling_model.py --evaluate --model-path models/compelling_scene_classifier
"""

import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    compute_metrics,
    save_model_with_metadata,
    load_model_with_metadata,
    ProgressTracker,
    create_dummy_dataset
)


class CompellingSceneClassifier(nn.Module):
    """
    Binary classifier for compelling vs non-compelling scenes.
    
    Takes VideoMAE embeddings and predicts if a scene is compelling
    (worthy of being a movieclips highlight).
    """
    
    def __init__(self, input_dim=768, hidden_dims=[512, 256], num_classes=2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


def load_training_data(data_dir, use_dummy=False):
    """Load training and validation data."""
    data_dir = Path(data_dir)
    
    if use_dummy:
        print("📦 Creating dummy data for testing...")
        dummy_dir = create_dummy_dataset(num_samples=100, save_dir=str(data_dir))
        data_dir = Path(dummy_dir)
    
    # Load features and labels
    train_features = np.load(data_dir / 'features.npy' if use_dummy else data_dir / 'train_features.npy')
    train_labels = np.load(data_dir / 'labels.npy' if use_dummy else data_dir / 'train_labels.npy')
    
    if not use_dummy and (data_dir / 'val_features.npy').exists():
        val_features = np.load(data_dir / 'val_features.npy')
        val_labels = np.load(data_dir / 'val_labels.npy')
    else:
        # Create validation split
        val_size = int(len(train_features) * 0.2)
        val_features = train_features[-val_size:]
        val_labels = train_labels[-val_size:]
        train_features = train_features[:-val_size]
        train_labels = train_labels[:-val_size]
    
    print(f"📊 Data loaded:")
    print(f"   Train: {len(train_features)} samples")
    print(f"   Val: {len(val_features)} samples")
    print(f"   Feature dim: {train_features.shape[1]}")
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.LongTensor(train_labels)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(val_features),
        torch.LongTensor(val_labels)
    )
    
    return train_dataset, val_dataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in tqdm(dataloader, desc="Training", leave=False):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # Compute additional metrics
    all_predictions = np.vstack(all_predictions)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    
    return avg_loss, accuracy, metrics


def train_model(args):
    """Main training loop."""
    print("=" * 60)
    print("🚀 TRAINING COMPELLING SCENE CLASSIFIER")
    print("=" * 60)
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"🖥️  Device: {device}\n")
    
    # Load data
    train_dataset, val_dataset = load_training_data(args.data_dir, use_dummy=args.use_dummy)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Get feature dimension from data
    feature_dim = train_dataset[0][0].shape[0]
    
    # Create model
    print("🤖 Creating model...")
    model = CompellingSceneClassifier(
        input_dim=feature_dim,
        hidden_dims=args.hidden_dims,
        num_classes=2
    )
    model.to(device)
    
    print(f"   Input dim: {feature_dim}")
    print(f"   Hidden dims: {args.hidden_dims}")
    print(f"   Output: 2 classes (compelling vs non-compelling)")
    print()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print("🏋️  Training...")
    print()
    
    tracker = ProgressTracker(args.num_epochs)
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss, val_acc, metrics = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track progress
        tracker.update(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
        print()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"✅ New best model! Val Acc: {val_acc:.4f}")
            print()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("=" * 60)
    print("📊 FINAL EVALUATION")
    print("=" * 60)
    
    val_loss, val_acc, metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / 'model.pt')
    
    # Save training info
    training_info = {
        'model_type': 'CompellingSceneClassifier',
        'feature_dim': feature_dim,
        'hidden_dims': args.hidden_dims,
        'num_classes': 2,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'best_val_acc': best_val_acc,
        'final_metrics': {
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'auc': float(metrics['auc'])
        }
    }
    
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Save training history
    tracker.save(output_dir / 'training_history.json')
    
    print(f"\n✅ Model saved to: {output_dir}")
    print()
    
    print("=" * 60)
    print("🎯 NEXT STEP")
    print("=" * 60)
    print("\nUse your trained model:")
    print(f"   python scripts/4_ensemble_scene_finder.py \\")
    print(f"       --video /path/to/movie.mp4 \\")
    print(f"       --compelling-model {output_dir}")
    print()


def evaluate_model(args):
    """Evaluate an existing model."""
    print("=" * 60)
    print("📊 EVALUATING MODEL")
    print("=" * 60)
    print()
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load training info
    with open(model_path / 'training_info.json', 'r') as f:
        training_info = json.load(f)
    
    print("Model Info:")
    print(f"   Type: {training_info['model_type']}")
    print(f"   Feature dim: {training_info['feature_dim']}")
    print(f"   Hidden dims: {training_info['hidden_dims']}")
    print(f"   Trained epochs: {training_info['num_epochs']}")
    print(f"   Best val acc: {training_info['best_val_acc']:.4f}")
    print()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    
    model = CompellingSceneClassifier(
        input_dim=training_info['feature_dim'],
        hidden_dims=training_info['hidden_dims'],
        num_classes=training_info['num_classes']
    )
    model.load_state_dict(torch.load(model_path / 'model.pt', map_location=device))
    model.to(device)
    
    print("✅ Model loaded successfully")
    print()
    
    # Load test data
    _, val_dataset = load_training_data(args.data_dir, use_dummy=args.use_dummy)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, metrics = evaluate(model, val_loader, criterion, device)
    
    print("Evaluation Results:")
    print(f"   Accuracy: {val_acc:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1 Score: {metrics['f1']:.4f}")
    print(f"   AUC: {metrics['auc']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Train compelling scene classifier'
    )
    
    # Mode
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate existing model instead of training'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/compelling_scene_classifier',
        help='Path to existing model (for evaluation)'
    )
    
    # Data
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/training_data',
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--use-dummy',
        action='store_true',
        help='Use dummy data for testing'
    )
    
    # Model architecture
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[512, 256],
        help='Hidden layer dimensions'
    )
    
    # Training
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=15,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        default=True,
        help='Use GPU if available'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/compelling_scene_classifier',
        help='Output directory for trained model'
    )
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_model(args)
    else:
        train_model(args)


if __name__ == '__main__':
    main()
