#!/usr/bin/env python3
"""
Create Enhanced Training Data
==============================

Creates a larger, more realistic training dataset using metadata-driven
synthetic features. This allows training without downloading 250GB of videos.

The synthetic features are designed to simulate VideoMAE embeddings with
realistic patterns based on clip metadata.
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import csv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import load_condensed_movies_metadata, create_training_split


def create_synthetic_features_from_metadata(clip_metadata, descriptions, movie_info, num_samples=5000):
    """
    Create synthetic features that simulate VideoMAE embeddings.
    
    Uses metadata (descriptions, duration, movie info) to create more
    realistic synthetic features than pure random noise.
    """
    print(f"🎨 Creating {num_samples} synthetic training samples...")
    print("   (Based on CondensedMovies metadata patterns)")
    print()
    
    positive_samples = []
    
    # Use actual clips from metadata as positive examples
    available_clips = clip_metadata[:num_samples] if len(clip_metadata) > num_samples else clip_metadata
    
    print(f"📊 Processing {len(available_clips)} clips from metadata...")
    
    for i, clip in enumerate(available_clips):
        if i % 500 == 0:
            print(f"   Processed {i}/{len(available_clips)} clips...")
        
        videoid = clip.get('videoid', '')
        imdbid = clip.get('imdbid', '')
        clip_name = clip.get('clip_name', '')
        
        # Get description
        desc_data = descriptions.get(videoid, {})
        description = desc_data.get('description', '')
        
        # Get movie genre
        movie_data = movie_info.get(imdbid, {})
        genre = movie_data.get('genre', '[]')
        
        # Create feature vector (768-dim like VideoMAE)
        features = create_feature_from_metadata(
            clip_name=clip_name,
            description=description,
            genre=genre
        )
        
        positive_samples.append({
            'features': features,
            'label': 1,  # Compelling
            'videoid': videoid,
            'clip_name': clip_name,
            'imdbid': imdbid
        })
    
    print(f"   ✅ Created {len(positive_samples)} positive samples")
    
    # Create negative samples (synthetic non-compelling segments)
    print(f"\n🔄 Creating {len(positive_samples)} negative samples...")
    
    negative_samples = []
    for i in range(len(positive_samples)):
        if i % 500 == 0:
            print(f"   Created {i}/{len(positive_samples)} negative samples...")
        
        # Negative samples have lower variance and different patterns
        features = np.random.randn(768).astype(np.float32) * 0.5  # Lower magnitude
        
        negative_samples.append({
            'features': features,
            'label': 0,  # Not compelling
            'videoid': f'synthetic_neg_{i}',
            'clip_name': 'Random segment',
            'imdbid': 'synthetic'
        })
    
    print(f"   ✅ Created {len(negative_samples)} negative samples")
    
    return positive_samples, negative_samples


def create_feature_from_metadata(clip_name, description, genre):
    """
    Create a 768-dim feature vector based on metadata.
    
    This simulates VideoMAE embeddings with patterns influenced by:
    - Clip name (indicates scene type)
    - Description (story context)
    - Genre (movie type)
    """
    # Base random features
    features = np.random.randn(768).astype(np.float32)
    
    # Adjust based on clip name patterns
    clip_lower = clip_name.lower()
    
    # Action/intense scenes - higher magnitude
    if any(word in clip_lower for word in ['battle', 'fight', 'chase', 'explosion', 'attack', 'war']):
        features *= 1.5
        features[:200] += np.random.randn(200) * 0.5  # Boost early features
    
    # Emotional scenes - different pattern
    elif any(word in clip_lower for word in ['love', 'death', 'farewell', 'goodbye', 'reunion', 'kiss']):
        features *= 1.2
        features[200:400] += np.random.randn(200) * 0.4
    
    # Dialogue/dramatic scenes
    elif any(word in clip_lower for word in ['speech', 'confrontation', 'confession', 'reveal', 'truth']):
        features[400:600] += np.random.randn(200) * 0.3
    
    # Musical/performance
    elif any(word in clip_lower for word in ['sing', 'dance', 'performance', 'music']):
        features *= 1.3
        features[600:768] += np.random.randn(168) * 0.5
    
    # Comedy
    elif any(word in clip_lower for word in ['funny', 'laugh', 'joke', 'comedy']):
        features *= 1.1
        features[::2] += np.random.randn(384) * 0.2  # Alternating pattern
    
    # Adjust based on description keywords
    if description:
        desc_lower = description.lower()
        
        # Suspense/tension
        if any(word in desc_lower for word in ['tension', 'suspense', 'dramatic', 'intense']):
            features *= 1.15
        
        # Climax/peak moments
        if any(word in desc_lower for word in ['climax', 'final', 'ultimate', 'crucial']):
            features *= 1.25
    
    # Genre influence
    if 'Action' in genre:
        features[:256] *= 1.1
    elif 'Drama' in genre:
        features[256:512] *= 1.1
    elif 'Comedy' in genre:
        features[512:768] *= 1.1
    
    return features


def main():
    parser = argparse.ArgumentParser(
        description='Create enhanced training data from metadata'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5000,
        help='Number of samples to create (will be doubled with negatives)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/enhanced_training',
        help='Output directory'
    )
    
    parser.add_argument(
        '--condensed-movies-dir',
        type=str,
        default='../datasets/CondensedMovies',
        help='Path to CondensedMovies dataset'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎨 CREATE ENHANCED TRAINING DATA")
    print("=" * 60)
    print()
    
    # Load metadata
    print("📚 Loading CondensedMovies metadata...")
    metadata = load_condensed_movies_metadata(args.condensed_movies_dir)
    
    clips = metadata.get('clips', [])
    descriptions = metadata.get('descriptions', {})
    movie_info = metadata.get('movie_info', {})
    
    print(f"   ✅ Loaded metadata")
    print(f"   Total clips available: {len(clips)}")
    print()
    
    # Create synthetic features
    positive_samples, negative_samples = create_synthetic_features_from_metadata(
        clips, descriptions, movie_info, num_samples=args.num_samples
    )
    
    # Combine
    all_samples = positive_samples + negative_samples
    
    # Extract features and labels
    all_features = np.array([s['features'] for s in all_samples])
    all_labels = np.array([s['label'] for s in all_samples])
    
    print(f"\n📊 Total dataset: {len(all_samples)} samples")
    print(f"   Positive (compelling): {len(positive_samples)}")
    print(f"   Negative (not compelling): {len(negative_samples)}")
    print()
    
    # Shuffle
    print("🔀 Shuffling dataset...")
    indices = np.random.permutation(len(all_samples))
    all_features = all_features[indices]
    all_labels = all_labels[indices]
    shuffled_samples = [all_samples[i] for i in indices]
    
    # Split train/val
    val_size = int(len(all_features) * args.val_split)
    split_idx = len(all_features) - val_size
    
    train_features = all_features[:split_idx]
    train_labels = all_labels[:split_idx]
    
    val_features = all_features[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"   Train: {len(train_features)} samples")
    print(f"   Val: {len(val_features)} samples")
    print()
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving to {output_dir}...")
    
    np.save(output_dir / 'train_features.npy', train_features)
    np.save(output_dir / 'train_labels.npy', train_labels)
    np.save(output_dir / 'val_features.npy', val_features)
    np.save(output_dir / 'val_labels.npy', val_labels)
    
    # Save metadata
    metadata = {
        'num_train': len(train_features),
        'num_val': len(val_features),
        'num_positive': int(np.sum(all_labels)),
        'num_negative': int(len(all_labels) - np.sum(all_labels)),
        'feature_dim': train_features.shape[1],
        'split_ratio': 1 - args.val_split,
        'source': 'CondensedMovies metadata + synthetic',
        'num_clips_used': len(positive_samples)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   ✅ Saved all files")
    print()
    
    print("=" * 60)
    print("✅ ENHANCED TRAINING DATA READY")
    print("=" * 60)
    print()
    print(f"Dataset: {output_dir}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Train: {len(train_features)}")
    print(f"Val: {len(val_features)}")
    print(f"Feature dim: {train_features.shape[1]}")
    print()
    print("Next step:")
    print(f"   python scripts/3_train_compelling_model.py --data-dir {output_dir}")
    print()


if __name__ == '__main__':
    main()
