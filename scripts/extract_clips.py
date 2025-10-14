#!/usr/bin/env python3
"""
Extract Video Clips
===================
Extract the top compelling scenes from the analyzed video.
"""

import json
import sys
from pathlib import Path
import subprocess

def extract_clip_opencv(input_video, output_video, start_time, duration):
    """Extract clip using OpenCV (no audio - use moviepy or ffmpeg instead)."""
    print("   ⚠️  OpenCV doesn't preserve audio - skipping")
    return False


def extract_clip_moviepy(input_video, output_video, start_time, duration):
    """Extract clip using moviepy."""
    try:
        from moviepy import VideoFileClip
        
        video = VideoFileClip(input_video)
        clip = video.subclipped(start_time, start_time + duration)
        clip.write_videofile(output_video, codec='libx264', audio_codec='aac', logger=None)
        video.close()
        
        return True
    except Exception as e:
        print(f"   MoviePy error: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract video clips')
    parser.add_argument('--results', type=str, default='output/analysis/analysis_results.json')
    parser.add_argument('--output-dir', type=str, default='output/compelling_clips')
    parser.add_argument('--method', type=str, choices=['opencv', 'moviepy', 'auto'], default='auto')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("✂️  EXTRACTING COMPELLING CLIPS")
    print("=" * 60)
    print()
    
    # Load results
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("\nRun analysis first:")
        print("   python scripts/simple_scene_finder.py --video videoplayback.mp4")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    video_path = results['video']
    top_segments = results['top_segments']
    
    print(f"📹 Video: {Path(video_path).name}")
    print(f"📊 Extracting {len(top_segments)} clips")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try extraction methods
    method = args.method
    
    for i, segment in enumerate(top_segments, 1):
        rank = segment['rank']
        start_time = segment['start_time']
        duration = segment['duration']
        score = segment['ensemble_score']
        
        output_file = output_dir / f"clip_{rank}_score_{score:.3f}.mp4"
        
        print(f"Extracting Clip #{rank}:")
        print(f"   Time: {start_time:.1f}s - {start_time + duration:.1f}s")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Score: {score:.3f}")
        
        success = False
        
        if method == 'auto' or method == 'opencv':
            print(f"   Method: OpenCV...")
            try:
                success = extract_clip_opencv(video_path, str(output_file), start_time, duration)
                if success:
                    print(f"   ✅ Saved: {output_file.name}")
            except Exception as e:
                print(f"   ❌ OpenCV failed: {e}")
        
        if not success and (method == 'auto' or method == 'moviepy'):
            print(f"   Method: MoviePy...")
            try:
                success = extract_clip_moviepy(video_path, str(output_file), start_time, duration)
                if success:
                    print(f"   ✅ Saved: {output_file.name}")
            except Exception as e:
                print(f"   ❌ MoviePy failed: {e}")
        
        if not success:
            print(f"   ⚠️  Could not extract clip {rank}")
        
        print()
    
    print("=" * 60)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nClips saved to: {output_dir}")
    print()
    
    # List extracted clips
    clips = list(output_dir.glob('*.mp4'))
    if clips:
        print(f"📁 Extracted {len(clips)} clips:")
        for clip in sorted(clips):
            size_mb = clip.stat().st_size / (1024 * 1024)
            print(f"   {clip.name} ({size_mb:.1f} MB)")
    
    print()


if __name__ == '__main__':
    main()
