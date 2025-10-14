#!/usr/bin/env python3
"""
Step 1: Prepare CondensedMovies Dataset
========================================

Downloads and prepares the CondensedMovies dataset for training.

Options:
- Download metadata (CSV files) - Required
- Download pre-computed features - Recommended (20GB)
- Download videos - Optional (250GB)
- Create dummy data - For testing

Usage:
    # Quick test
    python 1_prepare_condensed_movies.py --dummy
    
    # Recommended: metadata + features
    python 1_prepare_condensed_movies.py --features-only
    
    # Full: everything
    python 1_prepare_condensed_movies.py --download-videos
"""

import argparse
import os
import sys
from pathlib import Path
import json
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import load_condensed_movies_metadata, create_dummy_dataset


def prepare_dummy_data(args):
    """Create dummy data for testing."""
    print("🎯 Creating dummy training data...")
    print("   (For testing only - not real CondensedMovies data)\n")
    
    output_dir = Path(args.output_dir) / 'dummy_training'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy features
    create_dummy_dataset(
        num_samples=args.dummy_samples,
        save_dir=str(output_dir)
    )
    
    print("\n✅ Dummy data created!")
    print(f"   Location: {output_dir}")
    print(f"   Samples: {args.dummy_samples}")
    print("\nNext step:")
    print("   python scripts/3_train_compelling_model.py --use-dummy")


def check_dataset_exists(base_dir):
    """Check if CondensedMovies dataset exists."""
    dataset_dir = Path(base_dir)
    metadata_dir = dataset_dir / 'data' / 'metadata'
    
    if not metadata_dir.exists():
        return False
    
    required_files = ['clips.csv', 'movies.csv', 'descriptions.csv', 'durations.csv']
    for filename in required_files:
        if not (metadata_dir / filename).exists():
            return False
    
    return True


def prepare_metadata(args):
    """Prepare metadata from existing CondensedMovies dataset."""
    print("📊 Preparing CondensedMovies metadata...")
    
    dataset_dir = Path(args.condensed_movies_dir)
    
    if not check_dataset_exists(dataset_dir):
        print("\n❌ CondensedMovies dataset not found!")
        print(f"   Expected location: {dataset_dir}")
        print("\nThe dataset should already be in your datasets/ folder.")
        print("Check: datasets/CondensedMovies/data/metadata/")
        print("\nIf missing, you need to:")
        print("1. cd datasets/CondensedMovies/data_prep/")
        print("2. Edit config.json")
        print("3. Run: python download.py")
        return False
    
    # Load metadata
    print("   Loading metadata files...")
    metadata = load_condensed_movies_metadata(dataset_dir)
    
    print(f"\n✅ Metadata loaded successfully!")
    print(f"   Movies: {len(metadata.get('movies', {}))}")
    print(f"   Clips: {len(metadata.get('clips', []))}")
    print(f"   Descriptions: {len(metadata.get('descriptions', {}))}")
    
    # Save summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'num_movies': len(metadata.get('movies', {})),
        'num_clips': len(metadata.get('clips', [])),
        'dataset_path': str(dataset_dir),
        'status': 'metadata_ready'
    }
    
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return True


def prepare_features(args):
    """Prepare pre-computed features."""
    print("\n📦 Checking for pre-computed features...")
    
    dataset_dir = Path(args.condensed_movies_dir)
    features_dir = dataset_dir / 'data' / 'features'
    
    if features_dir.exists():
        print(f"✅ Features directory found: {features_dir}")
        
        # Count feature files
        feature_files = list(features_dir.glob('*.npy')) + list(features_dir.glob('*.pt'))
        print(f"   Feature files: {len(feature_files)}")
        
        if len(feature_files) > 0:
            return True
        else:
            print("   ⚠️ Feature directory exists but is empty")
    
    print("\n⚠️ Pre-computed features not found!")
    print("   You'll need to either:")
    print("   1. Download features using CondensedMovies download script")
    print("   2. Extract features yourself (see step 2)")
    print("\nTo download features:")
    print("   cd datasets/CondensedMovies/data_prep/")
    print("   Edit config.json (set 'features': true)")
    print("   python download.py")
    
    return False


def prepare_videos(args):
    """Check for videos (optional)."""
    print("\n🎥 Checking for videos...")
    
    dataset_dir = Path(args.condensed_movies_dir)
    videos_dir = dataset_dir / 'data' / 'videos'
    
    if videos_dir.exists():
        video_files = list(videos_dir.glob('*.mp4')) + list(videos_dir.glob('*.avi'))
        print(f"✅ Videos found: {len(video_files)}")
        
        if len(video_files) > 0:
            return True
    
    print("⚠️ Videos not found (optional)")
    print("   You can train using pre-computed features without videos.")
    print("   To download videos (250GB):")
    print("   cd datasets/CondensedMovies/data_prep/")
    print("   Edit config.json (set 'src': true)")
    print("   python download.py")
    
    return False


def create_training_config(args):
    """Create training configuration file."""
    print("\n⚙️ Creating training configuration...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'dataset': {
            'name': 'CondensedMovies',
            'base_dir': args.condensed_movies_dir,
            'num_clips': 34000,  # Approximate
            'split_ratio': 0.8,
            'split_by': 'movie'  # No movie appears in both train/val
        },
        'features': {
            'use_precomputed': args.features_only or args.use_precomputed,
            'feature_dim': 768,  # VideoMAE base
            'num_frames': 16
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 15,
            'learning_rate': 0.0001,
            'optimizer': 'adam',
            'mode': 'binary'  # compelling vs non-compelling
        },
        'model': {
            'backbone': 'MCG-NJU/videomae-base',
            'num_classes': 2,
            'freeze_backbone': False
        }
    }
    
    config_file = output_dir / 'training_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved: {config_file}")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Prepare CondensedMovies dataset for training'
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
        default='data',
        help='Output directory for prepared data'
    )
    
    parser.add_argument(
        '--dummy',
        action='store_true',
        help='Create dummy data for testing'
    )
    
    parser.add_argument(
        '--dummy-samples',
        type=int,
        default=100,
        help='Number of dummy samples'
    )
    
    parser.add_argument(
        '--features-only',
        action='store_true',
        help='Only use pre-computed features (no video download)'
    )
    
    parser.add_argument(
        '--download-videos',
        action='store_true',
        help='Download full videos (250GB)'
    )
    
    parser.add_argument(
        '--use-precomputed',
        action='store_true',
        help='Use pre-computed features from dataset'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📦 STEP 1: PREPARE CONDENSEDMOVIES DATASET")
    print("=" * 60)
    print()
    
    # Dummy mode
    if args.dummy:
        prepare_dummy_data(args)
        return
    
    # Check metadata
    print("Step 1.1: Metadata")
    print("-" * 60)
    metadata_ok = prepare_metadata(args)
    
    if not metadata_ok:
        print("\n❌ Setup failed - metadata not found")
        print("\nPlease ensure CondensedMovies dataset is downloaded:")
        print("   cd datasets/CondensedMovies/data_prep/")
        print("   python download.py")
        return
    
    # Check features
    print("\n\nStep 1.2: Features")
    print("-" * 60)
    features_ok = prepare_features(args)
    
    # Check videos (optional)
    print("\n\nStep 1.3: Videos (Optional)")
    print("-" * 60)
    videos_ok = prepare_videos(args)
    
    # Create config
    config = create_training_config(args)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PREPARATION SUMMARY")
    print("=" * 60)
    print(f"✅ Metadata: Ready")
    print(f"{'✅' if features_ok else '⚠️'} Features: {'Ready' if features_ok else 'Not found (will extract)'}")
    print(f"{'✅' if videos_ok else '⚠️'} Videos: {'Available' if videos_ok else 'Not available (optional)'}")
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS")
    print("=" * 60)
    
    if features_ok:
        print("\n✅ You're ready to train!")
        print("\nRecommended path (use pre-computed features):")
        print("   python scripts/2_extract_features.py --use-precomputed")
        print("   python scripts/3_train_compelling_model.py")
    elif videos_ok:
        print("\n✅ You have videos!")
        print("\nExtract features:")
        print("   python scripts/2_extract_features.py")
        print("\nThen train:")
        print("   python scripts/3_train_compelling_model.py")
    else:
        print("\n⚠️ You need either features or videos")
        print("\nOption 1 (Recommended): Download pre-computed features")
        print("   cd datasets/CondensedMovies/data_prep/")
        print("   Edit config.json: set 'features': true")
        print("   python download.py")
        print("\nOption 2: Download videos and extract features")
        print("   cd datasets/CondensedMovies/data_prep/")
        print("   Edit config.json: set 'src': true")
        print("   python download.py")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
