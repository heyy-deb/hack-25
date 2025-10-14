#!/usr/bin/env python3
"""
Download CondensedMovies Videos with Authentication
====================================================

Fixes YouTube bot detection by using browser cookies.

Usage:
    # Test with 10 videos
    python download_with_auth.py --num-videos 10 --browser chrome
    
    # Download 500 videos for training
    python download_with_auth.py --num-videos 500 --browser chrome
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path
import csv
import json

sys.path.append(str(Path(__file__).parent))
from utils import load_condensed_movies_metadata


def download_video_with_cookies(url, output_path, browser='chrome', timeout=120):
    """Download single video using browser cookies."""
    cmd = [
        'yt-dlp',
        '--cookies-from-browser', browser,
        '--no-playlist',
        '--no-warnings',
        '-f', 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        '--merge-output-format', 'mp4',
        '-o', str(output_path),
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Download CondensedMovies videos with authentication'
    )
    
    parser.add_argument(
        '--num-videos',
        type=int,
        default=50,
        help='Number of videos to download'
    )
    
    parser.add_argument(
        '--browser',
        type=str,
        default='chrome',
        choices=['chrome', 'firefox', 'safari'],
        help='Browser to extract cookies from (must be logged into YouTube)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../datasets/CondensedMovies/data/videos',
        help='Output directory for videos'
    )
    
    parser.add_argument(
        '--delay',
        type=int,
        default=2,
        help='Delay between downloads (seconds)'
    )
    
    parser.add_argument(
        '--condensed-movies-dir',
        type=str,
        default='../datasets/CondensedMovies',
        help='CondensedMovies dataset directory'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip videos that already exist'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📥 DOWNLOAD CONDENSEDMOVIES VIDEOS WITH AUTH")
    print("=" * 60)
    print()
    
    # Check yt-dlp
    try:
        result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
        print(f"✅ yt-dlp version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ yt-dlp not found!")
        print("\nInstall with:")
        print("   pip install yt-dlp")
        return
    
    print(f"🌐 Using {args.browser} browser cookies")
    print(f"⏱️  Delay: {args.delay}s between downloads")
    print()
    
    # Load metadata
    print("📚 Loading CondensedMovies metadata...")
    metadata = load_condensed_movies_metadata(args.condensed_movies_dir)
    clips = metadata.get('clips', [])
    
    print(f"   Total clips available: {len(clips)}")
    print()
    
    # Select diverse subset
    print(f"📝 Selecting {args.num_videos} clips...")
    
    selected_clips = []
    movies_used = set()
    
    # Stratified sampling for diversity
    for clip in clips:
        if len(selected_clips) >= args.num_videos:
            break
        
        imdbid = clip.get('imdbid')
        # Ensure diversity across movies
        if imdbid not in movies_used or len(selected_clips) < args.num_videos // 2:
            selected_clips.append(clip)
            movies_used.add(imdbid)
    
    print(f"   Selected {len(selected_clips)} clips from {len(movies_used)} movies")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download videos
    print("🎬 Starting downloads...")
    print(f"   Output: {output_dir}")
    print()
    print(f"⚠️  IMPORTANT: Make sure you're logged into YouTube in {args.browser}!")
    print()
    print("🚀 Starting automatic download...")
    print()
    
    successful = []
    failed = []
    skipped = []
    
    for i, clip in enumerate(selected_clips, 1):
        video_id = clip['videoid']
        url = f"https://www.youtube.com/watch?v={video_id}"
        output_file = output_dir / f"{video_id}.mp4"
        
        # Skip if exists
        if args.skip_existing and output_file.exists():
            skipped.append(clip)
            print(f"[{i}/{len(selected_clips)}] ⏭️  Skipped {video_id} (already exists)")
            continue
        
        print(f"[{i}/{len(selected_clips)}] Downloading {video_id}")
        print(f"   Movie: {clip.get('title', 'Unknown')}")
        print(f"   Scene: {clip.get('clip_name', 'Unknown')}")
        
        success, error_msg = download_video_with_cookies(url, output_file, args.browser)
        
        if success:
            successful.append(clip)
            size_mb = output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0
            print(f"   ✅ Success ({size_mb:.1f} MB)")
        else:
            failed.append({'clip': clip, 'error': error_msg})
            print(f"   ❌ Failed: {error_msg[:100]}")
        
        print()
        
        # Progress update
        if i % 10 == 0:
            print("-" * 60)
            print(f"Progress: {i}/{len(selected_clips)}")
            print(f"Success: {len(successful)} | Failed: {len(failed)} | Skipped: {len(skipped)}")
            print("-" * 60)
            print()
        
        # Delay
        if i < len(selected_clips):
            time.sleep(args.delay)
    
    # Final summary
    print()
    print("=" * 60)
    print("📊 DOWNLOAD COMPLETE")
    print("=" * 60)
    print()
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏭️  Skipped: {len(skipped)}")
    print(f"📁 Location: {output_dir}")
    print()
    
    success_rate = len(successful) / (len(successful) + len(failed)) * 100 if (len(successful) + len(failed)) > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    print()
    
    # Save metadata
    if successful:
        metadata_file = output_dir / 'downloaded_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'num_downloaded': len(successful),
                'num_failed': len(failed),
                'num_skipped': len(skipped),
                'success_rate': success_rate,
                'clips': successful
            }, f, indent=2)
        
        print(f"📄 Metadata saved: {metadata_file}")
        print()
    
    # Save failed list for retry
    if failed:
        failed_file = output_dir / 'failed_downloads.txt'
        with open(failed_file, 'w') as f:
            for item in failed:
                f.write(f"{item['clip']['videoid']}\t{item['error'][:50]}\n")
        
        print(f"📝 Failed list saved: {failed_file}")
        print()
    
    # Next steps
    if len(successful) > 0:
        print("=" * 60)
        print("🎯 NEXT STEPS")
        print("=" * 60)
        print()
        print(f"✅ Downloaded {len(successful)} videos!")
        print()
        print("Now extract features and train:")
        print()
        print("cd ../../condensed_movies_integration")
        print()
        print("python scripts/2_extract_features.py \\")
        print(f"    --video-dir {output_dir} \\")
        print("    --output-dir data/real_features")
        print()
        print("python scripts/3_train_compelling_model.py \\")
        print("    --data-dir data/real_features")
        print()
    else:
        print("=" * 60)
        print("⚠️  NO VIDEOS DOWNLOADED")
        print("=" * 60)
        print()
        print("Try these solutions:")
        print()
        print("1. Make sure you're logged into YouTube in your browser")
        print("2. Try a different browser: --browser firefox")
        print("3. Export cookies manually (see FIX_YOUTUBE_DOWNLOAD.md)")
        print("4. Use CondensedMovies Challenge pre-computed features")
        print("5. Continue with metadata-based model (already working!)")
        print()


if __name__ == '__main__':
    main()
