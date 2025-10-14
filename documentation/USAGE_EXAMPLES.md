# 🎬 Usage Examples

## Quick Reference Commands

### 1. Basic Usage (Balanced)
Find compelling scenes with equal weight to viral and story significance:

```bash
python scripts/4_ensemble_scene_finder.py \
    --video /path/to/movie.mp4 \
    --output output/clips
```

**Result:** Top 5 clips that are both viral-worthy AND compelling

---

### 2. Social Media Focus (Viral-Heavy)
Optimize for Instagram/TikTok engagement:

```bash
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --weight-viral 0.7 \
    --weight-compelling 0.3 \
    --top-n 10 \
    --output output/viral_clips
```

**Best for:** Instagram Reels, TikTok, YouTube Shorts

---

### 3. Movie Marketing (Balanced)
Perfect for trailers and promotional content:

```bash
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --weight-viral 0.5 \
    --weight-compelling 0.5 \
    --min-duration 90 \
    --max-duration 150 \
    --output output/promo_clips
```

**Best for:** Movie trailers, TV spots, promotional materials

---

### 4. Story-Focused (Compelling-Heavy)
Prioritize narrative significance and emotional impact:

```bash
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --weight-viral 0.3 \
    --weight-compelling 0.7 \
    --min-duration 60 \
    --max-duration 180 \
    --output output/story_clips
```

**Best for:** Film festival trailers, art house content, story highlights

---

### 5. Quick Test (3 Clips)
Fast test with just top 3 clips:

```bash
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --top-n 3 \
    --output output/test
```

---

## Real-World Scenarios

### Scenario 1: Content Creator (Instagram)
**Goal:** Create engaging Instagram Reels from popular movies

```bash
# Extract viral moments
python scripts/4_ensemble_scene_finder.py \
    --video avengers_endgame.mp4 \
    --weight-viral 0.8 \
    --weight-compelling 0.2 \
    --min-duration 30 \
    --max-duration 60 \
    --top-n 15 \
    --output output/instagram_reels

# Result: 15 highly viral clips (30-60 seconds each)
# Post to Instagram Reels with trending audio
```

---

### Scenario 2: Movie Studio Marketing
**Goal:** Create promotional assets for new movie release

```bash
# Extract trailer-worthy moments
python scripts/4_ensemble_scene_finder.py \
    --video new_blockbuster.mp4 \
    --weight-viral 0.5 \
    --weight-compelling 0.5 \
    --min-duration 90 \
    --max-duration 120 \
    --top-n 8 \
    --output output/marketing_assets

# Use clips for:
# - Social media teasers
# - TV spots
# - Press kit materials
```

---

### Scenario 3: Film Festival Submission
**Goal:** Create a compelling trailer that showcases story

```bash
# Extract emotionally significant moments
python scripts/4_ensemble_scene_finder.py \
    --video indie_film.mp4 \
    --weight-viral 0.2 \
    --weight-compelling 0.8 \
    --min-duration 60 \
    --max-duration 150 \
    --top-n 5 \
    --output output/festival_trailer

# Result: Story-driven clips that represent the film's themes
```

---

### Scenario 4: YouTube Channel
**Goal:** Create "Best Movie Moments" compilation

```bash
# Process multiple movies
for movie in movies/*.mp4; do
    python scripts/4_ensemble_scene_finder.py \
        --video "$movie" \
        --weight-viral 0.6 \
        --weight-compelling 0.4 \
        --top-n 3 \
        --output "output/$(basename "$movie" .mp4)"
done

# Combine all clips into compilation video
```

---

### Scenario 5: A/B Testing
**Goal:** Compare different weights to see what works best

```bash
# Test 1: Viral-focused
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --weight-viral 0.8 \
    --weight-compelling 0.2 \
    --output output/test_viral

# Test 2: Balanced
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --weight-viral 0.5 \
    --weight-compelling 0.5 \
    --output output/test_balanced

# Test 3: Story-focused
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --weight-viral 0.2 \
    --weight-compelling 0.8 \
    --output output/test_story

# Post all three to social media and track engagement
# Use winning strategy for future content
```

---

## Genre-Specific Recommendations

### Action Movies
```bash
python scripts/4_ensemble_scene_finder.py \
    --video action_movie.mp4 \
    --weight-viral 0.6 \
    --weight-compelling 0.4
```
**Why:** Action benefits from viral-worthy explosive moments

### Drama
```bash
python scripts/4_ensemble_scene_finder.py \
    --video drama.mp4 \
    --weight-viral 0.3 \
    --weight-compelling 0.7
```
**Why:** Drama needs emotional depth and story context

### Comedy
```bash
python scripts/4_ensemble_scene_finder.py \
    --video comedy.mp4 \
    --weight-viral 0.7 \
    --weight-compelling 0.3
```
**Why:** Comedy works best with laugh-out-loud viral moments

### Horror
```bash
python scripts/4_ensemble_scene_finder.py \
    --video horror.mp4 \
    --weight-viral 0.5 \
    --weight-compelling 0.5
```
**Why:** Balance jump scares (viral) with suspense (compelling)

---

## Advanced Usage

### Custom Scene Detection
Adjust scene detection sensitivity:

```bash
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --scene-threshold 30.0  # Higher = fewer scene cuts
```

### Specific Duration Range
Extract only clips of specific length:

```bash
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --min-duration 45 \
    --max-duration 75  # Perfect for 60-second Instagram Reels
```

### Extract Many Clips
Get more options to choose from:

```bash
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --top-n 20  # Extract top 20 clips
```

### CPU-Only Mode
For systems without GPU:

```bash
python scripts/4_ensemble_scene_finder.py \
    --video movie.mp4 \
    --no-use-gpu
```
*Note: Will be slower*

---

## Interpreting Results

### Understanding Scores

Each clip gets three scores:

1. **Viral Score (0-1):** Social media engagement potential
   - 0.8+ = Extremely viral-worthy
   - 0.6-0.8 = Good viral potential
   - 0.4-0.6 = Moderate
   - <0.4 = Low viral potential

2. **Compelling Score (0-1):** Story significance
   - 0.8+ = Highly significant moment
   - 0.6-0.8 = Important scene
   - 0.4-0.6 = Moderate importance
   - <0.4 = Less significant

3. **Ensemble Score (0-1):** Combined weighted score
   - This is what clips are ranked by
   - Depends on your chosen weights

### Example Output

```
Clip 1: movie_clip_1_score_0.847.mp4
   Time: 01:23:45 - 01:25:12
   Viral: 0.892 | Compelling: 0.801
   Ensemble: 0.847
```

**Interpretation:**
- High viral score (0.892) = Very shareable
- High compelling score (0.801) = Story-significant
- High ensemble (0.847) = Perfect clip for both metrics!

---

## Tips & Best Practices

### 1. Start with Balanced Weights
```bash
--weight-viral 0.5 --weight-compelling 0.5
```
Then adjust based on results and your goals

### 2. Test on Short Clips First
Don't process a 3-hour movie immediately. Test on:
- Movie trailers
- First 30 minutes
- Sample segments

### 3. Compare Results
Run with different weights and compare:
- Which clips get more engagement?
- Which better represent the movie?
- Which feel more "right" to you?

### 4. Consider Your Audience
- **Young/TikTok:** Higher viral weight (0.7-0.8)
- **Film buffs:** Higher compelling weight (0.6-0.8)
- **General:** Balanced (0.5-0.5)

### 5. Iterate Based on Feedback
Track engagement on posted clips:
- High engagement? → Increase viral weight
- Good comments? → Balance is working
- "Where's this from?" → Compelling weight working

---

## Troubleshooting Common Issues

### Issue: All clips look similar
**Solution:** Increase scene detection threshold
```bash
--scene-threshold 35.0  # More distinct scenes
```

### Issue: Clips too short/long
**Solution:** Adjust duration range
```bash
--min-duration 90 --max-duration 120
```

### Issue: Not finding best moments
**Solution:** Try pure compelling mode
```bash
--weight-viral 0.0 --weight-compelling 1.0
```

### Issue: Clips not engaging enough
**Solution:** Try pure viral mode
```bash
--weight-viral 1.0 --weight-compelling 0.0
```

---

## Batch Processing

### Process Multiple Movies

```bash
#!/bin/bash
# process_all_movies.sh

for movie in /path/to/movies/*.mp4; do
    basename=$(basename "$movie" .mp4)
    
    echo "Processing: $basename"
    
    python scripts/4_ensemble_scene_finder.py \
        --video "$movie" \
        --weight-viral 0.6 \
        --weight-compelling 0.4 \
        --top-n 5 \
        --output "output/${basename}_clips"
    
    echo "Completed: $basename"
    echo ""
done

echo "All movies processed!"
```

---

## Integration with Editing Tools

### Export for DaVinci Resolve
Clips are exported as standard MP4 files that can be imported directly into DaVinci Resolve.

### Export for Adobe Premiere
1. Extract clips using the tool
2. Import clips into Premiere project
3. Use `results.json` for metadata (scores, timestamps)

### Export for CapCut
1. Extract clips
2. Import to CapCut mobile app
3. Add captions, effects, trending audio
4. Post to social media

---

## Performance Optimization

### For Faster Processing
```bash
# Reduce frame sampling
--num-frames 8  # Default is 16

# Use GPU
--use-gpu  # Make sure CUDA is available

# Process fewer segments
--top-n 5  # Only extract top 5
```

### For Better Quality
```bash
# Increase frame sampling
--num-frames 32  # More frames = better analysis

# Lower scene threshold
--scene-threshold 25.0  # More granular scenes
```

---

## Next Steps

1. **Experiment:** Try different weights and see what works
2. **Track Results:** Monitor engagement on posted clips
3. **Refine:** Adjust weights based on performance
4. **Scale:** Once dialed in, batch process your library
5. **Automate:** Create scripts for your specific workflow

---

**Questions?** Check the README.md and SETUP_GUIDE.md

**Good luck finding compelling scenes! 🎬✨**
