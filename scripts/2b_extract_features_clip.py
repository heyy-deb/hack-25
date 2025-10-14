#!/usr/bin/env python3
"""
Step 2B: Extract Features using CLIP
=====================================

Uses CLIP (Contrastive Language-Image Pre-training) to extract multimodal features
that understand both visual content AND text descriptions.

CLIP is better than VideoMAE because:
- Understands vision + language jointly
- Can use scene descriptions from CondensedMovies metadata
- Faster inference (no temporal modeling overhead)
- Pre-trained on massive image-text pairs

Usage:
    python 2b_extract_features_clip.py --video-dir data/real_videos --max-videos 100
"""

import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch
import clip
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from utils import load_condensed_movies_metadata, get_video_info, create_training_split


class CLIPFeatureExtractor:
    """Extract features using CLIP vision encoder + text descriptions."""
    
    def __init__(self, model_name='ViT-B/32', device=None):
        print(f"🤖 Loading CLIP model: {model_name}")
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"   Device: {self.device}")
        
        # Load CLIP
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        print("   ✅ Model loaded")
    
    def extract_frames_from_video(self, video_path, num_frames=8):
        """Extract uniformly sampled frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames
    
    def extract_visual_features(self, video_path, num_frames=8):
        """Extract visual features from video frames using CLIP vision encoder."""
        frames = self.extract_frames_from_video(video_path, num_frames)
        
        if len(frames) == 0:
            return None
        
        # Preprocess frames
        images = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
        
        # Extract features
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            # Average pool across frames
            video_features = image_features.mean(dim=0)
            # Normalize
            video_features = video_features / video_features.norm()
        
        return video_features.cpu().numpy()
    
    def extract_text_features(self, text):
        """Extract text features using CLIP text encoder."""
        if not text or text.strip() == '':
            text = "a video scene"
        
        # Tokenize
        text_input = clip.tokenize([text]).to(self.device)
        
        # Extract features
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features = text_features / text_features.norm()
        
        return text_features.cpu().numpy().flatten()
    
    def extract_multimodal_features(self, video_path, scene_description=None, num_frames=8):
        """
        Extract combined visual + text features.
        
        This leverages CLIP's joint embedding space where vision and text 
        features are aligned, making the model understand "compelling scenes"
        better than vision-only models.
        """
        # Visual features
        visual_features = self.extract_visual_features(video_path, num_frames)
        
        if visual_features is None:
            return None
        
        # Text features (if description available)
        if scene_description:
            text_features = self.extract_text_features(scene_description)
            # Concatenate visual + text features
            features = np.concatenate([visual_features, text_features])
        else:
            features = visual_features
        
        return features


def main():
    parser = argparse.ArgumentParser(description='Extract CLIP features from CondensedMovies videos')
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/real_videos',
        help='Directory containing downloaded videos'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/clip_features',
        help='Output directory for features'
    )
    
    parser.add_argument(
        '--condensed-movies-dir',
        type=str,
        default='../datasets/CondensedMovies',
        help='CondensedMovies dataset directory'
    )
    
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum number of videos to process (for testing)'
    )
    
    parser.add_argument(
        '--num-frames',
        type=int,
        default=8,
        help='Number of frames to extract per video'
    )
    
    parser.add_argument(
        '--use-text',
        action='store_true',
        help='Use text descriptions (multimodal features)'
    )
    
    parser.add_argument(
        '--clip-model',
        type=str,
        default='ViT-B/32',
        choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
        help='CLIP model variant'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 STEP 2B: EXTRACT CLIP FEATURES")
    print("=" * 60)
    print()
    
    # Load metadata
    print("🎬 Loading CondensedMovies metadata...")
    metadata = load_condensed_movies_metadata(args.condensed_movies_dir)
    clips_metadata = metadata.get('clips', [])
    print(f"   Clips in metadata: {len(clips_metadata)}")
    
    # Find available videos
    video_dir = Path(args.video_dir)
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.webm')) + list(video_dir.glob('*.mkv'))
    print(f"   Video files found: {len(video_files)}")
    
    # Create video_id -> metadata mapping
    video_id_to_metadata = {clip['videoid']: clip for clip in clips_metadata}
    
    # Filter videos that have metadata
    videos_to_process = []
    for video_file in video_files:
        video_id = video_file.stem.split('.')[0]  # Remove extension
        if video_id in video_id_to_metadata:
            videos_to_process.append((video_file, video_id_to_metadata[video_id]))
    
    if args.max_videos:
        videos_to_process = videos_to_process[:args.max_videos]
    
    print(f"   Videos with metadata: {len(videos_to_process)}")
    print()
    
    # Initialize CLIP
    extractor = CLIPFeatureExtractor(model_name=args.clip_model)
    
    # Feature dimension
    if args.use_text:
        feature_dim = 1024  # 512 (visual) + 512 (text) for ViT-B/32
        print(f"📊 Using multimodal features (visual + text): {feature_dim}D")
    else:
        feature_dim = 512  # visual only for ViT-B/32
        print(f"📊 Using visual features only: {feature_dim}D")
    print()
    
    # Extract features
    print("📥 Extracting features...")
    features_list = []
    labels_list = []
    metadata_list = []
    
    for video_file, clip_metadata in tqdm(videos_to_process, desc="Processing videos"):
        try:
            # Get scene description
            scene_desc = None
            if args.use_text:
                clip_name = clip_metadata.get('clip_name', '')
                title = clip_metadata.get('title', '')
                scene_desc = f"{title}: {clip_name}"
            
            # Extract features
            features = extractor.extract_multimodal_features(
                video_file, 
                scene_description=scene_desc,
                num_frames=args.num_frames
            )
            
            if features is not None:
                features_list.append(features)
                labels_list.append(1)  # All CondensedMovies clips are positive (compelling)
                metadata_list.append({
                    'video_id': clip_metadata['videoid'],
                    'title': clip_metadata.get('title', ''),
                    'clip_name': clip_metadata.get('clip_name', ''),
                    'video_file': str(video_file)
                })
        
        except Exception as e:
            print(f"   ⚠️  Error processing {video_file.name}: {e}")
            continue
    
    print()
    print(f"✅ Extracted features from {len(features_list)} clips")
    
    # Convert to numpy arrays
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"   Features shape: {features.shape}")
    print()
    
    # Create negative samples
    print("🔄 Creating negative samples...")
    print("   (Augmented versions with noise/perturbations)")
    
    num_negatives = len(features)
    negative_features = []
    
    for i in range(num_negatives):
        # Create negative by adding noise and slight rotation to embedding space
        noise = np.random.normal(0, 0.15, features[i].shape)
        negative = features[i] + noise
        # Normalize
        negative = negative / np.linalg.norm(negative)
        negative_features.append(negative)
    
    negative_features = np.array(negative_features)
    negative_labels = np.zeros(num_negatives)
    
    print(f"   ✅ Created {num_negatives} negative samples (augmented)")
    print("   Note: For best results, use real random movie segments")
    print()
    
    # Combine positive and negative
    print("🔧 Combining training data...")
    all_features = np.vstack([features, negative_features])
    all_labels = np.concatenate([labels, negative_labels])
    
    # Create train/val split
    total_samples = len(all_features)
    indices = np.random.permutation(total_samples)
    train_size = int(0.8 * total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_features = all_features[train_indices]
    train_labels = all_labels[train_indices]
    val_features = all_features[val_indices]
    val_labels = all_labels[val_indices]
    
    print()
    print("✅ Training data prepared!")
    print(f"   Train samples: {len(train_features)}")
    print(f"   Val samples: {len(val_features)}")
    print(f"   Positive: {len(features)}")
    print(f"   Negative: {num_negatives}")
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    training_dir = output_dir / 'training_data'
    training_dir.mkdir(exist_ok=True)
    
    np.save(training_dir / 'train_features.npy', train_features)
    np.save(training_dir / 'train_labels.npy', train_labels)
    np.save(training_dir / 'val_features.npy', val_features)
    np.save(training_dir / 'val_labels.npy', val_labels)
    
    # Save metadata
    with open(training_dir / 'metadata.json', 'w') as f:
        json.dump({
            'num_videos': len(features_list),
            'feature_dim': feature_dim,
            'num_frames': args.num_frames,
            'clip_model': args.clip_model,
            'use_text': args.use_text,
            'train_samples': len(train_features),
            'val_samples': len(val_features),
            'clips_metadata': metadata_list
        }, f, indent=2)
    
    print(f"   Saved to: {training_dir}")
    print()
    
    print("=" * 60)
    print("🎯 NEXT STEP")
    print("=" * 60)
    print()
    print("Train the model:")
    print(f"   python scripts/3_train_compelling_model.py --data-dir {training_dir}")
    print()


if __name__ == '__main__':
    main()
