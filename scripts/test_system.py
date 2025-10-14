#!/usr/bin/env python3
"""
Quick System Test
=================

Tests the trained compelling scene classifier on dummy features.
This verifies the system is working without requiring full video processing.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import load_model_with_metadata


class CompellingSceneClassifier(nn.Module):
    """Compelling scene classifier (same as training script)."""
    
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


def test_model():
    """Test the trained model on synthetic features."""
    print("=" * 60)
    print("🧪 TESTING COMPELLING SCENE CLASSIFIER")
    print("=" * 60)
    print()
    
    # Load model
    model_path = Path('models/compelling_scene_classifier')
    
    if not model_path.exists():
        print("❌ Model not found. Train first:")
        print("   python scripts/3_train_compelling_model.py --use-dummy")
        return
    
    print("📥 Loading trained model...")
    
    with open(model_path / 'training_info.json', 'r') as f:
        training_info = json.load(f)
    
    print(f"   Model trained on {training_info.get('num_samples', 'N/A')} samples")
    print(f"   Best validation accuracy: {training_info['best_val_acc']:.4f}")
    print()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CompellingSceneClassifier(
        input_dim=training_info['feature_dim'],
        hidden_dims=training_info['hidden_dims'],
        num_classes=2
    )
    model.load_state_dict(torch.load(model_path / 'model.pt', map_location=device))
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    print()
    
    # Create synthetic test features
    print("🎬 Testing on synthetic movie scene features...")
    print()
    
    test_scenes = [
        {"name": "Dramatic dialogue scene", "features": np.random.randn(768) * 1.5},
        {"name": "Action sequence", "features": np.random.randn(768) * 2.0},
        {"name": "Quiet emotional moment", "features": np.random.randn(768) * 0.8},
        {"name": "Epic battle", "features": np.random.randn(768) * 2.5},
        {"name": "Random filler", "features": np.random.randn(768) * 0.5},
    ]
    
    print("Scoring test scenes:")
    print("-" * 60)
    
    for i, scene in enumerate(test_scenes, 1):
        features = torch.FloatTensor(scene['features']).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            compelling_score = probs[0, 1].item()
            prediction = "COMPELLING" if compelling_score > 0.5 else "Not compelling"
        
        print(f"\n{i}. {scene['name']}")
        print(f"   Compelling Score: {compelling_score:.3f}")
        print(f"   Prediction: {prediction}")
        
        # Simulate viral score (would come from your existing viral model)
        viral_score = np.random.uniform(0.4, 0.9)
        
        # Ensemble score (50/50 blend)
        ensemble_score = 0.5 * viral_score + 0.5 * compelling_score
        
        print(f"   Viral Score (simulated): {viral_score:.3f}")
        print(f"   Ensemble Score: {ensemble_score:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ SYSTEM TEST COMPLETE")
    print("=" * 60)
    print()
    
    print("The model is working! It can now score movie scenes.")
    print()
    print("Next steps:")
    print("1. Train on real CondensedMovies data for better accuracy")
    print("2. Use with full video processing to find compelling scenes")
    print()
    print("For full video processing:")
    print("   python scripts/4_ensemble_scene_finder.py --video movie.mp4")
    print()
    
    # Show model architecture
    print("Model Architecture:")
    print("-" * 60)
    print(f"Input: {training_info['feature_dim']}-dim features (from VideoMAE)")
    print(f"Hidden layers: {training_info['hidden_dims']}")
    print(f"Output: 2 classes (compelling vs non-compelling)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Show ensemble strategy
    print("Ensemble Strategy:")
    print("-" * 60)
    print("ensemble_score = viral_weight × viral_score +")
    print("                 compelling_weight × compelling_score")
    print()
    print("Recommended weights:")
    print("  • Social media: viral=0.7, compelling=0.3")
    print("  • Marketing: viral=0.5, compelling=0.5")
    print("  • Story-focused: viral=0.3, compelling=0.7")
    print()


if __name__ == '__main__':
    test_model()
