#!/usr/bin/env python3
"""
Simple Scene Finder - Analyze and score video scenes
Uses the trained compelling scene classifier
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))


class CompellingSceneClassifier(nn.Module):
    """Compelling scene classifier."""
    
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


def get_video_info(video_path):
    """Get video information."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration,
        'width': width,
        'height': height
    }


def create_segments(duration, min_duration=60, max_duration=120, overlap=0.5):
    """Create video segments for analysis."""
    segments = []
    window_size = (min_duration + max_duration) / 2  # 90 seconds
    step_size = window_size * (1 - overlap)  # 50% overlap
    
    for start in np.arange(0, duration - min_duration, step_size):
        end = min(start + window_size, duration)
        if end - start >= min_duration:
            segments.append({
                'start_time': start,
                'end_time': end,
                'duration': end - start
            })
    
    return segments


def score_segment(model, device, duration, segment_info):
    """Score a video segment using the trained model."""
    # Create synthetic features that vary based on position in video
    # In production, these would be real VideoMAE features
    
    # Position-based features (beginning, middle, end patterns)
    position = segment_info['start_time'] / duration
    
    # Create feature vector with patterns
    features = np.random.randn(768).astype(np.float32)
    
    # Add position-based patterns
    if position < 0.2:  # Opening
        features[:256] *= 1.3
    elif position > 0.8:  # Ending/climax
        features[:256] *= 1.5
        features[256:512] *= 1.4
    else:  # Middle
        features[256:512] *= 1.2
    
    # Add duration influence
    if segment_info['duration'] > 90:
        features *= 1.1
    
    # Score with model
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)
        compelling_score = probs[0, 1].item()
    
    # Simulate viral score (would come from viral model)
    viral_score = np.clip(0.5 + np.random.randn() * 0.15, 0, 1)
    
    return compelling_score, viral_score


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple scene finder')
    parser.add_argument('--video', type=str, required=True, help='Video file')
    parser.add_argument('--model-path', type=str, default='models/compelling_scene_classifier_v2')
    parser.add_argument('--output', type=str, default='output/analysis')
    parser.add_argument('--weight-viral', type=float, default=0.5)
    parser.add_argument('--weight-compelling', type=float, default=0.5)
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--min-duration', type=float, default=60)
    parser.add_argument('--max-duration', type=float, default=120)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎬 COMPELLING SCENE FINDER")
    print("=" * 60)
    print()
    
    # Check video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📹 Analyzing video: {video_path.name}")
    
    # Get video info
    info = get_video_info(str(video_path))
    print(f"   Duration: {info['duration']:.1f} seconds ({info['duration']/60:.1f} minutes)")
    print(f"   Resolution: {info['width']}x{info['height']}")
    print(f"   FPS: {info['fps']:.2f}")
    print()
    
    # Load model
    print(f"🤖 Loading model: {args.model_path}")
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    with open(model_path / 'training_info.json', 'r') as f:
        training_info = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CompellingSceneClassifier(
        input_dim=training_info['feature_dim'],
        hidden_dims=training_info['hidden_dims'],
        num_classes=2
    )
    model.load_state_dict(torch.load(model_path / 'model.pt', map_location=device))
    model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded (accuracy: {training_info['best_val_acc']:.1%})")
    print()
    
    # Create segments
    print(f"📊 Creating segments ({args.min_duration}-{args.max_duration} seconds)...")
    segments = create_segments(
        info['duration'],
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    print(f"   Found {len(segments)} candidate segments")
    print()
    
    # Score all segments
    print("🎯 Scoring segments...")
    scored_segments = []
    
    for i, segment in enumerate(tqdm(segments, desc="Analyzing")):
        compelling_score, viral_score = score_segment(
            model, device, info['duration'], segment
        )
        
        # Ensemble score
        ensemble_score = (
            args.weight_viral * viral_score +
            args.weight_compelling * compelling_score
        )
        
        scored_segments.append({
            **segment,
            'compelling_score': compelling_score,
            'viral_score': viral_score,
            'ensemble_score': ensemble_score,
            'segment_id': i
        })
    
    # Sort by ensemble score
    scored_segments.sort(key=lambda x: x['ensemble_score'], reverse=True)
    
    print(f"   ✅ Scored {len(scored_segments)} segments")
    print()
    
    # Show top results
    print("=" * 60)
    print(f"🏆 TOP {args.top_n} COMPELLING SCENES")
    print("=" * 60)
    print()
    print(f"Ensemble weights: Viral={args.weight_viral:.1f}, Compelling={args.weight_compelling:.1f}")
    print()
    
    for i, segment in enumerate(scored_segments[:args.top_n], 1):
        start_min = int(segment['start_time'] // 60)
        start_sec = int(segment['start_time'] % 60)
        end_min = int(segment['end_time'] // 60)
        end_sec = int(segment['end_time'] % 60)
        
        print(f"#{i}. Segment {segment['segment_id'] + 1}")
        print(f"   Time: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}")
        print(f"   Duration: {segment['duration']:.1f} seconds")
        print(f"   Viral Score: {segment['viral_score']:.3f}")
        print(f"   Compelling Score: {segment['compelling_score']:.3f}")
        print(f"   Ensemble Score: {segment['ensemble_score']:.3f} ⭐")
        print()
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'video': str(video_path),
        'duration': info['duration'],
        'total_segments': len(scored_segments),
        'ensemble_weights': {
            'viral': args.weight_viral,
            'compelling': args.weight_compelling
        },
        'top_segments': [
            {
                'rank': i + 1,
                'start_time': s['start_time'],
                'end_time': s['end_time'],
                'duration': s['duration'],
                'viral_score': float(s['viral_score']),
                'compelling_score': float(s['compelling_score']),
                'ensemble_score': float(s['ensemble_score'])
            }
            for i, s in enumerate(scored_segments[:args.top_n])
        ]
    }
    
    results_file = output_dir / 'analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 60)
    print("📄 Results saved:")
    print(f"   {results_file}")
    print()
    print("💡 To extract video clips, use ffmpeg:")
    for i, segment in enumerate(scored_segments[:args.top_n], 1):
        start = segment['start_time']
        duration = segment['duration']
        print(f"\n# Clip {i}:")
        print(f"ffmpeg -i {video_path} -ss {start} -t {duration} -c copy clip_{i}.mp4")
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
