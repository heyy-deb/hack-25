#!/usr/bin/env python3
"""
Step 4: Ensemble Scene Finder
==============================

Combines TWO powerful models to find the best movie scenes:
1. Your existing viral clip finder (social media virality)
2. New CondensedMovies model (story significance)

The result: Scenes that are BOTH viral AND compelling!

Usage:
    # Basic usage
    python 4_ensemble_scene_finder.py --video /path/to/movie.mp4
    
    # Custom weights (more viral-focused)
    python 4_ensemble_scene_finder.py \
        --video movie.mp4 \
        --weight-viral 0.7 \
        --weight-compelling 0.3
    
    # Custom weights (more story-focused)
    python 4_ensemble_scene_finder.py \
        --video movie.mp4 \
        --weight-viral 0.3 \
        --weight-compelling 0.7
"""

import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    import cv2
    from moviepy import VideoFileClip
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nInstall with:")
    print("   pip install scenedetect moviepy transformers torch")
    sys.exit(1)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import extract_frames_uniform, get_video_info, format_time


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


class EnsembleSceneFinder:
    """
    Combines viral clip finder and compelling scene classifier.
    
    Finds scenes that are:
    - Viral-worthy (engaging, shareable, high retention)
    - Compelling (story-significant, emotional, memorable)
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
        
        print("🎬 ENSEMBLE SCENE FINDER")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Ensemble weights: Viral={args.weight_viral:.2f}, Compelling={args.weight_compelling:.2f}")
        print()
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load both viral and compelling models."""
        print("📥 Loading models...")
        
        # 1. Load VideoMAE for feature extraction
        print("   Loading VideoMAE...")
        self.processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
        self.videomae = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base')
        self.videomae.to(self.device)
        self.videomae.eval()
        
        # 2. Load compelling scene classifier
        if self.args.compelling_model and Path(self.args.compelling_model).exists():
            print(f"   Loading compelling model: {self.args.compelling_model}")
            
            model_path = Path(self.args.compelling_model)
            with open(model_path / 'training_info.json', 'r') as f:
                training_info = json.load(f)
            
            self.compelling_model = CompellingSceneClassifier(
                input_dim=training_info['feature_dim'],
                hidden_dims=training_info['hidden_dims'],
                num_classes=2
            )
            self.compelling_model.load_state_dict(
                torch.load(model_path / 'model.pt', map_location=self.device)
            )
            self.compelling_model.to(self.device)
            self.compelling_model.eval()
            
            self.has_compelling_model = True
            print(f"   ✅ Compelling model loaded (Val Acc: {training_info['best_val_acc']:.4f})")
        else:
            print("   ⚠️  No compelling model found - using viral model only")
            self.has_compelling_model = False
        
        # 3. Check for existing viral model (optional)
        viral_model_path = Path(self.args.viral_model) if self.args.viral_model else None
        if viral_model_path and viral_model_path.exists():
            print(f"   Loading viral model: {viral_model_path}")
            # Load your fine-tuned viral model here
            # For now, we'll use VideoMAE features + simple scoring
            self.has_viral_model = True
        else:
            print("   ℹ️  Using VideoMAE features for viral scoring")
            self.has_viral_model = False
        
        print("   ✅ All models loaded")
        print()
    
    def segment_video(self, video_path):
        """Segment video into candidate clips."""
        print("📹 Detecting scenes...")
        
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.args.scene_threshold))
        
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        video_manager.release()
        
        # Group scenes into clips of desired duration
        video_info = get_video_info(video_path)
        fps = video_info['fps']
        
        segments = []
        current_start = 0
        
        for i, (start_time, end_time) in enumerate(scene_list):
            duration = (end_time - start_time).get_seconds()
            
            if duration >= self.args.min_duration and duration <= self.args.max_duration:
                segments.append({
                    'start_time': start_time.get_seconds(),
                    'end_time': end_time.get_seconds(),
                    'duration': duration,
                    'segment_id': len(segments)
                })
        
        # Also create sliding window segments
        total_duration = video_info['duration']
        window_size = (self.args.min_duration + self.args.max_duration) / 2
        step_size = window_size * 0.5  # 50% overlap
        
        for start in np.arange(0, total_duration - window_size, step_size):
            end = min(start + window_size, total_duration)
            duration = end - start
            
            if duration >= self.args.min_duration:
                segments.append({
                    'start_time': start,
                    'end_time': end,
                    'duration': duration,
                    'segment_id': len(segments)
                })
        
        print(f"   Found {len(segments)} candidate segments")
        print()
        
        return segments
    
    def extract_features(self, video_path, start_time, end_time):
        """Extract features from video segment."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame
        
        # Sample frames uniformly
        num_frames = self.args.num_frames
        if total_frames < num_frames:
            frame_indices = list(range(start_frame, end_frame))
            frame_indices += [end_frame - 1] * (num_frames - len(frame_indices))
        else:
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) < num_frames:
            return None
        
        # Process with VideoMAE
        inputs = self.processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.videomae(**inputs, output_hidden_states=True)
            # Extract features from last hidden layer
            features = outputs.hidden_states[-1].mean(dim=1).squeeze()
        
        return features.cpu().numpy()
    
    def score_viral(self, features):
        """Score segment for viral potential."""
        # Simple scoring based on feature magnitudes
        # In production, use your fine-tuned viral model
        
        feature_energy = np.linalg.norm(features)
        feature_variance = np.var(features)
        feature_sparsity = np.sum(np.abs(features) > 0.1) / len(features)
        
        # Normalize to 0-1
        viral_score = np.tanh(feature_energy / 100) * 0.4
        viral_score += np.tanh(feature_variance * 10) * 0.3
        viral_score += feature_sparsity * 0.3
        
        return viral_score
    
    def score_compelling(self, features):
        """Score segment for compelling-ness."""
        if not self.has_compelling_model:
            return 0.5  # Neutral score
        
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.compelling_model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            compelling_score = probs[0, 1].item()  # Probability of "compelling" class
        
        return compelling_score
    
    def rank_segments(self, video_path, segments):
        """Rank all segments using ensemble scoring."""
        print("🎯 Scoring segments...")
        
        scored_segments = []
        
        for segment in tqdm(segments, desc="Analyzing"):
            # Extract features
            features = self.extract_features(
                video_path,
                segment['start_time'],
                segment['end_time']
            )
            
            if features is None:
                continue
            
            # Score with both models
            viral_score = self.score_viral(features)
            compelling_score = self.score_compelling(features)
            
            # Ensemble score
            ensemble_score = (
                self.args.weight_viral * viral_score +
                self.args.weight_compelling * compelling_score
            )
            
            scored_segments.append({
                **segment,
                'viral_score': viral_score,
                'compelling_score': compelling_score,
                'ensemble_score': ensemble_score,
                'features': features
            })
        
        # Sort by ensemble score
        scored_segments.sort(key=lambda x: x['ensemble_score'], reverse=True)
        
        print(f"   ✅ Scored {len(scored_segments)} segments")
        print()
        
        return scored_segments
    
    def extract_top_clips(self, video_path, segments):
        """Extract top N clips to files."""
        print(f"✂️  Extracting top {self.args.top_n} clips...")
        
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video = VideoFileClip(video_path)
        video_name = Path(video_path).stem
        
        extracted_clips = []
        
        for i, segment in enumerate(segments[:self.args.top_n]):
            clip_name = f"{video_name}_clip_{i+1}_score_{segment['ensemble_score']:.3f}.mp4"
            clip_path = output_dir / clip_name
            
            try:
                clip = video.subclipped(segment['start_time'], segment['end_time'])
                clip.write_videofile(
                    str(clip_path),
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=str(output_dir / f'temp_audio_{i}.m4a'),
                    remove_temp=True,
                    logger=None
                )
                
                extracted_clips.append({
                    'clip_path': str(clip_path),
                    **segment
                })
                
                print(f"   ✅ Clip {i+1}: {format_time(segment['start_time'])} - {format_time(segment['end_time'])}")
                print(f"      Viral: {segment['viral_score']:.3f} | Compelling: {segment['compelling_score']:.3f} | Ensemble: {segment['ensemble_score']:.3f}")
            
            except Exception as e:
                print(f"   ❌ Failed to extract clip {i+1}: {e}")
        
        video.close()
        
        return extracted_clips
    
    def save_results(self, clips, output_dir):
        """Save analysis results."""
        results = {
            'num_clips': len(clips),
            'ensemble_weights': {
                'viral': self.args.weight_viral,
                'compelling': self.args.weight_compelling
            },
            'clips': [
                {
                    'clip_path': c['clip_path'],
                    'start_time': c['start_time'],
                    'end_time': c['end_time'],
                    'duration': c['duration'],
                    'viral_score': float(c['viral_score']),
                    'compelling_score': float(c['compelling_score']),
                    'ensemble_score': float(c['ensemble_score'])
                }
                for c in clips
            ]
        }
        
        results_file = Path(output_dir) / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results saved: {results_file}")
    
    def process_video(self, video_path):
        """Main processing pipeline."""
        print("=" * 60)
        print(f"📽️  Processing: {video_path}")
        print("=" * 60)
        print()
        
        # Segment video
        segments = self.segment_video(video_path)
        
        # Rank segments
        scored_segments = self.rank_segments(video_path, segments)
        
        # Extract top clips
        clips = self.extract_top_clips(video_path, scored_segments)
        
        # Save results
        self.save_results(clips, self.args.output_dir)
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ PROCESSING COMPLETE")
        print("=" * 60)
        print(f"\nExtracted {len(clips)} clips to: {self.args.output_dir}")
        print("\nTop 3 clips:")
        for i, clip in enumerate(clips[:3]):
            print(f"\n{i+1}. {Path(clip['clip_path']).name}")
            print(f"   Time: {format_time(clip['start_time'])} - {format_time(clip['end_time'])}")
            print(f"   Viral: {clip['viral_score']:.3f} | Compelling: {clip['compelling_score']:.3f}")
            print(f"   Ensemble: {clip['ensemble_score']:.3f}")
        
        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Find compelling movie scenes using ensemble approach'
    )
    
    # Input
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to video file'
    )
    
    # Models
    parser.add_argument(
        '--compelling-model',
        type=str,
        default='models/compelling_scene_classifier',
        help='Path to trained compelling scene model'
    )
    
    parser.add_argument(
        '--viral-model',
        type=str,
        default='../datasets_and_models/videomae_viral_model',
        help='Path to trained viral clip model'
    )
    
    # Ensemble weights
    parser.add_argument(
        '--weight-viral',
        type=float,
        default=0.5,
        help='Weight for viral score (0-1)'
    )
    
    parser.add_argument(
        '--weight-compelling',
        type=float,
        default=0.5,
        help='Weight for compelling score (0-1)'
    )
    
    # Segmentation
    parser.add_argument(
        '--min-duration',
        type=float,
        default=60,
        help='Minimum clip duration (seconds)'
    )
    
    parser.add_argument(
        '--max-duration',
        type=float,
        default=120,
        help='Maximum clip duration (seconds)'
    )
    
    parser.add_argument(
        '--scene-threshold',
        type=float,
        default=27.0,
        help='Scene detection threshold'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/compelling_clips',
        help='Output directory for clips'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top clips to extract'
    )
    
    # Processing
    parser.add_argument(
        '--num-frames',
        type=int,
        default=16,
        help='Number of frames to sample per segment'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        default=True,
        help='Use GPU if available'
    )
    
    args = parser.parse_args()
    
    # Validate weights sum to 1
    total_weight = args.weight_viral + args.weight_compelling
    if abs(total_weight - 1.0) > 0.01:
        print(f"⚠️  Warning: Weights sum to {total_weight}, normalizing...")
        args.weight_viral /= total_weight
        args.weight_compelling /= total_weight
    
    # Check video exists
    if not Path(args.video).exists():
        print(f"❌ Video not found: {args.video}")
        return
    
    # Run ensemble finder
    finder = EnsembleSceneFinder(args)
    finder.process_video(args.video)


if __name__ == '__main__':
    main()
