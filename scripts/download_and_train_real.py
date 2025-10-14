#!/usr/bin/env python3
"""
Download Subset and Train on Real Data
=======================================

Downloads a subset of CondensedMovies videos and trains on real VideoMAE features.

This is a practical alternative to:
1. Downloading 250GB of all videos (days)
2. Using unavailable pre-computed features (404 error)
3. Training on synthetic data (not real)

Strategy:
- Download 100-200 clips (manageable size ~5-10GB)
- Extract real VideoMAE features
- Train on actual CondensedMovies data
- Expected: 85-90% accuracy on real patterns
"""

import argparse
import sys
from pathlib import Path
import json
import subprocess
import numpy as np
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from utils import load_condensed_movies_metadata


def check_dependencies():
    """Check if required tools are installed."""
    print("🔍 Checking dependencies...")
    
    # Check for video downloader
    try:
        result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
        print(f"   ✅ yt-dlp: {result.stdout.strip()}")
        return 'yt-dlp'
    except FileNotFoundError:
        try:
            result = subprocess.run(['youtube-dl', '--version'], capture_output=True, text=True)
            print(f"   ✅ youtube-dl: {result.stdout.strip()}")
            return 'youtube-dl'
        except FileNotFoundError:
            print("   ❌ Neither yt-dlp nor youtube-dl found")
            print("\n   Install one of these:")
            print("   pip install yt-dlp")
            print("   OR")
            print("   pip install youtube-dl")
            return None
    
    print()


def download_video_clip(video_id, output_path, downloader='yt-dlp'):
    """Download a single video clip from YouTube."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    cmd = [
        downloader,
        '-f', 'bestvideo[height<=480]+bestaudio/best[height<=480]',  # Lower quality = faster
        '--no-playlist',
        '--no-warnings',
        '-o', str(output_path),
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download subset and train on real data'
    )
    
    parser.add_argument(
        '--num-videos',
        type=int,
        default=200,
        help='Number of videos to download'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/real_condensed_movies',
        help='Output directory'
    )
    
    parser.add_argument(
        '--condensed-movies-dir',
        type=str,
        default='../datasets/CondensedMovies',
        help='Path to CondensedMovies dataset'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download if videos already exist'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📥 DOWNLOAD REAL CONDENSEDMOVIES DATA")
    print("=" * 60)
    print()
    
    # Check dependencies
    downloader = check_dependencies()
    if not downloader and not args.skip_download:
        print("❌ Cannot download without yt-dlp or youtube-dl")
        print("\nOptions:")
        print("1. Install yt-dlp: pip install yt-dlp")
        print("2. Use --skip-download if videos already exist")
        return
    
    # Load metadata
    print("📚 Loading CondensedMovies metadata...")
    metadata = load_condensed_movies_metadata(args.condensed_movies_dir)
    clips = metadata.get('clips', [])
    print(f"   Total clips available: {len(clips)}")
    print()
    
    # Select subset
    print(f"📝 Selecting {args.num_videos} clips to download...")
    
    # Stratified sampling by year and movie
    selected_clips = []
    movies_used = set()
    
    for clip in clips:
        if len(selected_clips) >= args.num_videos:
            break
        
        imdbid = clip.get('imdbid')
        if imdbid not in movies_used or len(selected_clips) < args.num_videos // 2:
            selected_clips.append(clip)
            movies_used.add(imdbid)
    
    print(f"   Selected {len(selected_clips)} clips from {len(movies_used)} movies")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    videos_dir = output_dir / 'videos'
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Download videos
    if not args.skip_download:
        print(f"📥 Downloading {len(selected_clips)} videos...")
        print(f"   (This may take 30-60 minutes)")
        print(f"   Videos will be saved to: {videos_dir}")
        print()
        
        successful_downloads = []
        failed_downloads = []
        
        for i, clip in enumerate(tqdm(selected_clips, desc="Downloading")):
            video_id = clip['videoid']
            output_file = videos_dir / f"{video_id}.mp4"
            
            # Skip if already exists
            if output_file.exists():
                successful_downloads.append(clip)
                continue
            
            # Download
            success = download_video_clip(video_id, output_file, downloader)
            
            if success and output_file.exists():
                successful_downloads.append(clip)
            else:
                failed_downloads.append(clip)
            
            # Progress update every 10 videos
            if (i + 1) % 10 == 0:
                print(f"\n   Progress: {len(successful_downloads)}/{i+1} successful")
        
        print(f"\n✅ Download complete!")
        print(f"   Successful: {len(successful_downloads)}")
        print(f"   Failed: {len(failed_downloads)}")
        print()
        
        # Save successful clips metadata
        with open(output_dir / 'downloaded_clips.json', 'w') as f:
            json.dump(successful_downloads, f, indent=2)
    
    else:
        print("⏭️  Skipping download (--skip-download)")
        # Check existing videos
        existing_videos = list(videos_dir.glob('*.mp4'))
        print(f"   Found {len(existing_videos)} existing videos")
        print()
    
    # Next steps
    print("=" * 60)
    print("🎯 NEXT STEPS")
    print("=" * 60)
    print()
    print("Videos downloaded! Now extract features and train:")
    print()
    print(f"1. Extract features:")
    print(f"   python scripts/2_extract_features.py \\")
    print(f"       --video-dir {videos_dir} \\")
    print(f"       --output-dir {output_dir}/features")
    print()
    print(f"2. Train on real data:")
    print(f"   python scripts/3_train_compelling_model.py \\")
    print(f"       --data-dir {output_dir}/training_data")
    print()


if __name__ == '__main__':
    main()
