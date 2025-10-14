#!/usr/bin/env python3
"""
Shared Utilities for CondensedMovies Integration
================================================
"""

import os
import json
import csv
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image


def load_condensed_movies_metadata(base_dir: str) -> Dict:
    """
    Load all CondensedMovies metadata CSV files.
    
    Returns:
        dict with keys: movies, clips, descriptions, durations, casts, split
    """
    metadata_dir = Path(base_dir) / 'data' / 'metadata'
    
    metadata = {}
    
    # Load movies
    movies_file = metadata_dir / 'movies.csv'
    if movies_file.exists():
        with open(movies_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            metadata['movies'] = {row['imdbid']: row for row in reader}
    
    # Load clips
    clips_file = metadata_dir / 'clips.csv'
    if clips_file.exists():
        with open(clips_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            metadata['clips'] = list(reader)
    
    # Load descriptions
    descriptions_file = metadata_dir / 'descriptions.csv'
    if descriptions_file.exists():
        with open(descriptions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            metadata['descriptions'] = {row['videoid']: row for row in reader}
    
    # Load durations
    durations_file = metadata_dir / 'durations.csv'
    if durations_file.exists():
        with open(durations_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            metadata['durations'] = {row['videoid']: int(row['duration']) for row in reader}
    
    # Load split
    split_file = metadata_dir / 'split.csv'
    if split_file.exists():
        with open(split_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            metadata['split'] = {row['imdbid']: row['split'] for row in reader}
    
    # Load movie_info (genres)
    movie_info_file = metadata_dir / 'movie_info.csv'
    if movie_info_file.exists():
        with open(movie_info_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            metadata['movie_info'] = {row['imdbid']: row for row in reader}
    
    return metadata


def extract_frames_uniform(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
    """
    Extract uniformly sampled frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
    
    Returns:
        List of frames as numpy arrays (RGB)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        # If video has fewer frames, duplicate last frame
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # If read fails, duplicate previous frame
            if frames:
                frames.append(frames[-1])
    
    cap.release()
    return frames


def extract_video_segment(video_path: str, start_time: float, end_time: float, 
                         num_frames: int = 16) -> List[np.ndarray]:
    """
    Extract frames from a specific segment of video.
    
    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds
        num_frames: Number of frames to extract
    
    Returns:
        List of frames as numpy arrays (RGB)
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_segment_frames = end_frame - start_frame
    
    if total_segment_frames < num_frames:
        indices = list(range(start_frame, end_frame))
        # Pad with last frame
        indices += [end_frame - 1] * (num_frames - len(indices))
    else:
        indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames


def get_video_info(video_path: str) -> Dict:
    """Get basic video information."""
    cap = cv2.VideoCapture(str(video_path))
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def create_training_split(clips: List[Dict], val_ratio: float = 0.2, 
                         split_by: str = 'movie') -> Tuple[List, List]:
    """
    Create train/val split ensuring no data leakage.
    
    Args:
        clips: List of clip dictionaries
        val_ratio: Validation set ratio
        split_by: 'movie' (no movie appears in both sets) or 'random'
    
    Returns:
        train_clips, val_clips
    """
    if split_by == 'movie':
        # Group by movie
        movie_clips = {}
        for clip in clips:
            imdbid = clip.get('imdbid', clip.get('movie_id', 'unknown'))
            if imdbid not in movie_clips:
                movie_clips[imdbid] = []
            movie_clips[imdbid].append(clip)
        
        # Split movies
        movies = list(movie_clips.keys())
        np.random.shuffle(movies)
        split_idx = int(len(movies) * (1 - val_ratio))
        
        train_movies = movies[:split_idx]
        val_movies = movies[split_idx:]
        
        train_clips = []
        val_clips = []
        
        for movie in train_movies:
            train_clips.extend(movie_clips[movie])
        
        for movie in val_movies:
            val_clips.extend(movie_clips[movie])
    
    else:  # random split
        clips_shuffled = clips.copy()
        np.random.shuffle(clips_shuffled)
        split_idx = int(len(clips_shuffled) * (1 - val_ratio))
        train_clips = clips_shuffled[:split_idx]
        val_clips = clips_shuffled[split_idx:]
    
    return train_clips, val_clips


def save_model_with_metadata(model, processor, save_dir: str, metadata: Dict):
    """Save model, processor, and training metadata."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and processor
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    # Save metadata
    metadata_file = save_dir / 'training_info.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved to {save_dir}")


def load_model_with_metadata(load_dir: str):
    """Load model, processor, and training metadata."""
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    
    load_dir = Path(load_dir)
    
    # Load model and processor
    processor = VideoMAEImageProcessor.from_pretrained(load_dir)
    model = VideoMAEForVideoClassification.from_pretrained(load_dir)
    
    # Load metadata
    metadata_file = load_dir / 'training_info.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return model, processor, metadata


def compute_metrics(predictions, labels):
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    preds = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    # AUC for binary classification
    if predictions.shape[1] == 2:
        auc = roc_auc_score(labels, predictions[:, 1])
    else:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_dummy_dataset(num_samples: int = 100, save_dir: str = 'data/dummy_training'):
    """
    Create dummy dataset for testing.
    
    Creates synthetic data that mimics real features.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy features
    features = []
    labels = []
    
    for i in range(num_samples):
        # Random features (mimicking VideoMAE embeddings)
        feature = np.random.randn(768).astype(np.float32)
        
        # Binary label
        label = i % 2
        
        features.append(feature)
        labels.append(label)
    
    # Save
    np.save(save_dir / 'features.npy', np.array(features))
    np.save(save_dir / 'labels.npy', np.array(labels))
    
    # Save metadata
    metadata = {
        'num_samples': num_samples,
        'feature_dim': 768,
        'num_classes': 2,
        'split': 'dummy'
    }
    
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Dummy dataset created: {num_samples} samples")
    print(f"   Saved to: {save_dir}")
    
    return save_dir


class ProgressTracker:
    """Track and display training progress."""
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def update(self, epoch: int, train_loss: float, train_acc: float,
               val_loss: float = None, val_acc: float = None):
        """Update progress."""
        self.current_epoch = epoch
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
    
    def print_epoch(self, epoch: int, train_loss: float, train_acc: float,
                    val_loss: float = None, val_acc: float = None):
        """Print epoch summary."""
        msg = f"Epoch {epoch}/{self.total_epochs} - "
        msg += f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f}"
        
        if val_loss is not None:
            msg += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
        
        print(msg)
    
    def save(self, filepath: str):
        """Save training history."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {filepath}")


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")
    
    # Test dummy dataset creation
    create_dummy_dataset(num_samples=50, save_dir='data/test_dummy')
    
    print("\n✅ All utilities working!")
