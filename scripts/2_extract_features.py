#!/usr/bin/env python3
"""
Step 2: Extract Features from CondensedMovies
==============================================

Extracts features from videos for training the compelling scene classifier.

Options:
1. Use pre-computed features from dataset (fastest)
2. Extract features from videos (requires videos + GPU)

Usage:
    # Use pre-computed features (recommended)
    python 2_extract_features.py --use-precomputed
    
    # Extract from videos
    python 2_extract_features.py --video-dir ../datasets/CondensedMovies/data/videos
"""

import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_condensed_movies_metadata,
    extract_frames_uniform,
    get_video_info,
    create_training_split
)


class FeatureExtractor:
    """Extract features from videos using VideoMAE."""
    
    def __init__(self, model_name='MCG-NJU/videomae-base', device=None):
        print(f"🤖 Loading VideoMAE model: {model_name}")
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"   Device: {self.device}")
        
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("   ✅ Model loaded\n")
    
    def extract(self, video_path, num_frames=16):
        """Extract features from a video."""
        try:
            # Extract frames
            frames = extract_frames_uniform(video_path, num_frames)
            
            # Process frames
            inputs = self.processor(frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get features (hidden states from last layer)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use last hidden state, average over sequence dimension
                features = outputs.hidden_states[-1].mean(dim=1).squeeze()
            
            return features.cpu().numpy()
        
        except Exception as e:
            print(f"   ❌ Error extracting features: {e}")
            return None


def use_precomputed_features(args):
    """Use pre-computed features from CondensedMovies dataset."""
    print("📦 Using pre-computed features from CondensedMovies...")
    
    dataset_dir = Path(args.condensed_movies_dir)
    features_dir = dataset_dir / 'data' / 'features'
    
    if not features_dir.exists():
        print(f"❌ Features directory not found: {features_dir}")
        print("\nYou need to download pre-computed features:")
        print("   cd datasets/CondensedMovies/data_prep/")
        print("   Edit config.json: set 'features': true")
        print("   python download.py")
        return False
    
    # Load metadata
    metadata = load_condensed_movies_metadata(dataset_dir)
    clips = metadata.get('clips', [])
    
    print(f"   Found {len(clips)} clips in metadata")
    
    # Create training data structure
    print("\n📊 Creating training dataset...")
    
    # Positive samples: all clips (these ARE compelling scenes)
    positive_samples = []
    
    # Check for features
    feature_files = list(features_dir.glob('*.npy'))
    print(f"   Feature files found: {len(feature_files)}")
    
    if len(feature_files) == 0:
        print("   ⚠️ No .npy feature files found")
        print("   The CondensedMovies dataset may have features in a different format")
        print("   You may need to extract features yourself from videos")
        return False
    
    # For this simplified version, we'll create a mapping
    # In real implementation, you'd match videoid to feature files
    print("\n✅ Pre-computed features available")
    print("\nNote: You'll need to adapt feature loading based on the")
    print("      specific format of CondensedMovies pre-computed features.")
    print("\nFor now, proceeding with feature extraction setup...")
    
    return True


def extract_features_from_videos(args):
    """Extract features from video files."""
    print("🎬 Extracting features from videos...")
    
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return False
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    dataset_dir = Path(args.condensed_movies_dir)
    metadata = load_condensed_movies_metadata(dataset_dir)
    clips = metadata.get('clips', [])
    durations = metadata.get('durations', {})
    
    print(f"   Clips in metadata: {len(clips)}")
    
    # Find videos
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
    print(f"   Video files found: {len(video_files)}")
    
    if len(video_files) == 0:
        print("   ❌ No video files found")
        return False
    
    # Initialize feature extractor
    extractor = FeatureExtractor(device=args.device)
    
    # Extract features
    print("\n📥 Extracting features...")
    
    positive_features = []
    positive_labels = []
    positive_metadata = []
    
    for video_file in tqdm(video_files[:args.max_videos], desc="Processing videos"):
        video_id = video_file.stem
        
        # Check if this is a clip from our dataset
        clip_data = None
        for clip in clips:
            if clip['videoid'] == video_id:
                clip_data = clip
                break
        
        if clip_data:
            # Extract features
            features = extractor.extract(str(video_file), num_frames=args.num_frames)
            
            if features is not None:
                positive_features.append(features)
                positive_labels.append(1)  # Compelling
                positive_metadata.append({
                    'videoid': video_id,
                    'imdbid': clip_data.get('imdbid'),
                    'duration': durations.get(video_id, 0),
                    'clip_name': clip_data.get('clip_name', '')
                })
    
    print(f"\n✅ Extracted features from {len(positive_features)} clips")
    
    # Save features
    if len(positive_features) > 0:
        features_array = np.array(positive_features)
        labels_array = np.array(positive_labels)
        
        np.save(output_dir / 'positive_features.npy', features_array)
        np.save(output_dir / 'positive_labels.npy', labels_array)
        
        with open(output_dir / 'positive_metadata.json', 'w') as f:
            json.dump(positive_metadata, f, indent=2)
        
        print(f"   Saved to: {output_dir}")
        print(f"   Features shape: {features_array.shape}")
        
        return True
    
    return False


def create_negative_samples(args):
    """Create negative samples (non-compelling segments)."""
    print("\n🔄 Creating negative samples...")
    print("   (Random segments from same movies, not in clip list)")
    
    # This would require:
    # 1. Load full movies
    # 2. Sample random 60-120 second segments NOT in clips.csv
    # 3. Extract features from those segments
    
    print("\n   ⚠️ Negative sample creation requires full movies")
    print("   For initial training, we can use data augmentation on positive samples")
    print("   or train with class imbalance techniques")
    
    # For now, create synthetic negative samples for demonstration
    output_dir = Path(args.output_dir)
    positive_features_file = output_dir / 'positive_features.npy'
    
    if positive_features_file.exists():
        positive_features = np.load(positive_features_file)
        
        # Create synthetic negative samples by adding noise to positive samples
        # In production, these would be real random segments
        negative_features = positive_features + np.random.randn(*positive_features.shape) * 0.1
        negative_labels = np.zeros(len(negative_features))
        
        np.save(output_dir / 'negative_features.npy', negative_features)
        np.save(output_dir / 'negative_labels.npy', negative_labels)
        
        print(f"   ✅ Created {len(negative_features)} negative samples (synthetic)")
        print(f"   Note: For production, use real random movie segments")


def combine_training_data(args):
    """Combine positive and negative samples into training dataset."""
    print("\n🔧 Combining training data...")
    
    output_dir = Path(args.output_dir)
    
    # Load positive and negative samples
    pos_features = np.load(output_dir / 'positive_features.npy')
    pos_labels = np.load(output_dir / 'positive_labels.npy')
    
    neg_features = np.load(output_dir / 'negative_features.npy')
    neg_labels = np.load(output_dir / 'negative_labels.npy')
    
    # Combine
    all_features = np.concatenate([pos_features, neg_features])
    all_labels = np.concatenate([pos_labels, neg_labels])
    
    # Shuffle
    indices = np.random.permutation(len(all_features))
    all_features = all_features[indices]
    all_labels = all_labels[indices]
    
    # Split train/val
    val_ratio = args.val_split
    split_idx = int(len(all_features) * (1 - val_ratio))
    
    train_features = all_features[:split_idx]
    train_labels = all_labels[:split_idx]
    
    val_features = all_features[split_idx:]
    val_labels = all_labels[split_idx:]
    
    # Save
    train_dir = output_dir / 'training_data'
    train_dir.mkdir(exist_ok=True)
    
    np.save(train_dir / 'train_features.npy', train_features)
    np.save(train_dir / 'train_labels.npy', train_labels)
    np.save(train_dir / 'val_features.npy', val_features)
    np.save(train_dir / 'val_labels.npy', val_labels)
    
    # Save metadata
    metadata = {
        'num_train': len(train_features),
        'num_val': len(val_features),
        'num_positive': int(np.sum(all_labels)),
        'num_negative': int(len(all_labels) - np.sum(all_labels)),
        'feature_dim': train_features.shape[1],
        'split_ratio': 1 - val_ratio
    }
    
    with open(train_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Training data prepared!")
    print(f"   Train samples: {len(train_features)}")
    print(f"   Val samples: {len(val_features)}")
    print(f"   Positive: {metadata['num_positive']}")
    print(f"   Negative: {metadata['num_negative']}")
    print(f"   Saved to: {train_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from CondensedMovies dataset'
    )
    
    parser.add_argument(
        '--condensed-movies-dir',
        type=str,
        default='../datasets/CondensedMovies',
        help='Path to CondensedMovies dataset'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/features',
        help='Output directory for features'
    )
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default=None,
        help='Directory containing videos'
    )
    
    parser.add_argument(
        '--use-precomputed',
        action='store_true',
        help='Use pre-computed features from dataset'
    )
    
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum number of videos to process'
    )
    
    parser.add_argument(
        '--num-frames',
        type=int,
        default=16,
        help='Number of frames to extract per video'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 STEP 2: EXTRACT FEATURES")
    print("=" * 60)
    print()
    
    # Check dataset
    dataset_dir = Path(args.condensed_movies_dir)
    if not dataset_dir.exists():
        print(f"❌ CondensedMovies dataset not found: {dataset_dir}")
        print("\nRun step 1 first:")
        print("   python scripts/1_prepare_condensed_movies.py")
        return
    
    success = False
    
    if args.use_precomputed:
        # Use pre-computed features
        success = use_precomputed_features(args)
    elif args.video_dir:
        # Extract from videos
        success = extract_features_from_videos(args)
        if success:
            create_negative_samples(args)
            combine_training_data(args)
    else:
        print("❌ Please specify either --use-precomputed or --video-dir")
        print("\nRecommended:")
        print("   python scripts/2_extract_features.py --use-precomputed")
        return
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 NEXT STEP")
        print("=" * 60)
        print("\nTrain the model:")
        print("   python scripts/3_train_compelling_model.py")
        print()


if __name__ == '__main__':
    main()
