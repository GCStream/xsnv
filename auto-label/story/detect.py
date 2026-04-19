#!/usr/bin/env python3
"""
Story Detection - Detect narrative flow in albums

Usage:
    python -m story.detect --album 12345
    python -m story.detect --all
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
from PIL import Image
from imagehash import phash
import requests
from datasets import load_dataset


# Default threshold - can be overridden via config
DEFAULT_MIN_FRAMES = 30
DEFAULT_KEY_FRAME_THRESHOLD = 0.7


def load_image(image_path: Path) -> Optional[Image.Image]:
    """Load and verify image."""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception:
        return None


def compute_image_hash(image_path: Path) -> Optional[str]:
    """Compute perceptual hash for image similarity."""
    try:
        img = load_image(image_path)
        if img is None:
            return None
        return str(phash(img))
    except Exception:
        return None


def compute_histogram(image_path: Path) -> Optional[np.ndarray]:
    """Compute color histogram for change detection."""
    try:
        img = load_image(image_path)
        if img is None:
            return None

        # Compute RGB histogram
        hist = []
        for i in range(3):
            channel_hist = img.histogram()[i*256:(i+1)*256]
            hist.extend(channel_hist)

        # Normalize
        hist = np.array(hist, dtype=float)
        hist = hist / (hist.sum() + 1e-7)
        return hist
    except Exception:
        return None


def cosine_similarity_hist(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Calculate cosine similarity between histograms."""
    dot = np.dot(hist1, hist2)
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def detect_outfit_change(histograms: List[np.ndarray], threshold: float = 0.85) -> List[int]:
    """Detect outfit changes based on color histogram changes."""
    changes = [0]  # First frame is always a key frame

    for i in range(1, len(histograms)):
        sim = cosine_similarity_hist(histograms[i-1], histograms[i])
        if sim < threshold:
            changes.append(i)

    return changes


def detect_pose_sequence(image_hashes: List[str], threshold: float = 0.9) -> List[int]:
    """Detect pose sequence based on image similarity."""
    key_frames = [0]

    for i in range(1, len(image_hashes)):
        if image_hashes[i] != image_hashes[i-1]:
            key_frames.append(i)

    return key_frames


def detect_location_change(histograms: List[np.ndarray], window: int = 5) -> List[int]:
    """Detect location changes (sustained histogram changes)."""
    key_frames = [0]

    for i in range(window, len(histograms)):
        # Check if histogram has been stable in recent window
        recent_sims = []
        for j in range(max(0, i - window), i):
            sim = cosine_similarity_hist(histograms[j], histograms[j+1])
            recent_sims.append(sim)

        avg_sim = np.mean(recent_sims)
        if avg_sim < 0.7:  # Location changed
            key_frames.append(i)

    return key_frames


def infer_story_type(key_frames: List[int], fill_count: int, total: int) -> str:
    """Infer story type from frame distribution."""
    if total < 10:
        return "random"

    filler_ratio = fill_count / total

    if filler_ratio > 0.7:
        return "random"
    elif len(key_frames) <= 3:
        return "minimal"
    elif len(key_frames) <= 8:
        return "outfit_progression"
    elif len(key_frames) <= 15:
        return "location_change"
    else:
        return "pose_series"


def analyze_album(album_dir: Path, min_frames: int = DEFAULT_MIN_FRAMES) -> Dict:
    """Analyze story in an album."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Sort by filename (numerically) - dataset images are shuffled
    def sort_key(path: Path) -> tuple:
        name = path.stem
        # Try numeric sort
        try:
            return (0, int(name))
        except ValueError:
            return (1, name)
    
    images = sorted(
        [f for f in album_dir.iterdir() if f.suffix.lower() in image_extensions],
        key=sort_key
    )

    if len(images) < min_frames:
        return {
            "album": album_dir.name,
            "story_type": "insufficient_frames",
            "error": f"Only {len(images)} images, need {min_frames}+",
        }

    print(f"Analyzing {len(images)} images in {album_dir.name}...")

    # Compute hashes and histograms
    image_hashes = []
    histograms = []
    metadata = []

    for img_path in images:
        img_hash = compute_image_hash(img_path)
        hist = compute_histogram(img_path)

        if img_hash:
            image_hashes.append(img_hash)
        if hist is not None:
            histograms.append(hist)

        # Basic metadata
        img = load_image(img_path)
        width, height = img.size if img else (0, 0)
        metadata.append({
            "path": str(img_path.name),
            "width": width,
            "height": height,
            "hash": img_hash,
        })

    if len(histograms) < 10:
        return {
            "album": album_dir.name,
            "story_type": "insufficient_data",
            "error": "Failed to compute sufficient features",
        }

    # Detect changes using multiple methods
    outfit_changes = detect_outfit_change(histograms, threshold=0.85)
    pose_frames = detect_pose_sequence(image_hashes, threshold=0.9)
    location_frames = detect_location_change(histograms, window=5)

    # Combine: take union of key frames
    all_key_frames = sorted(set(outfit_changes + pose_frames + location_frames))

    # Mark filler frames
    is_filler = []
    for i in range(len(metadata)):
        if i in all_key_frames:
            is_filler.append(False)
        else:
            is_filler.append(True)

    # Generate descriptions
    story_type = infer_story_type(all_key_frames, sum(is_filler), len(metadata))

    # Text descriptions (template - actual generation uses vLLM)
    text_en = _generate_text_en(story_type, len(metadata), all_key_frames)
    text_cn = _generate_text_cn(story_type, len(metadata), all_key_frames)

    result = {
        "album": album_dir.name,
        "image_count": len(images),
        "min_frames_required": min_frames,
        "story_type": story_type,
        "key_frames": all_key_frames,
        "key_frame_count": len(all_key_frames),
        "is_filler": is_filler,
        "filler_count": sum(is_filler),
        "detection": {
            "outfit_changes": outfit_changes,
            "pose_changes": pose_frames,
            "location_changes": location_frames,
        },
        "text_en": text_en,
        "text_cn": text_cn,
    }

    print(f"  -> story_type: {story_type}, key_frames: {len(all_key_frames)}, filler: {sum(is_filler)}")

    return result


def _generate_text_en(story_type: str, total: int, key_frames: List[int]) -> str:
    """Generate English text description (template)."""
    templates = {
        "outfit_progression": f"A photo series of {total} images showing model changing outfits. {len(key_frames)} key styling moments captured.",
        "location_change": f"A photo series of {total} images showing model in different locations. {len(key_frames)} distinctive scenes.",
        "pose_series": f"A photo series of {total} images showcasing various poses. {len(key_frames)} key pose variations.",
        "minimal": f"A photo series of {total} images with {len(key_frames)} major transformations.",
        "random": f"A collection of {total} images in varied settings and styles.",
    }
    return templates.get(story_type, f"A photo series with {total} images.")


def _generate_text_cn(story_type: str, total: int, key_frames: List[int]) -> str:
    """Generate Chinese text description (template)."""
    templates = {
        "outfit_progression": f"共{total}张图片展示模特换装过程，{len(key_frames)}个关键造型时刻。",
        "location_change": f"共{total}张图片展示不同场景，{len(key_frames)}个主要场景切换。",
        "pose_series": f"共{total}张图片展示多种姿势，{len(key_frames)}个关键姿势变化。",
        "minimal": f"共{total}张图片，{len(key_frames)}个主要变换。",
        "random": f"共{total}张 varied 图片集。",
    }
    return templates.get(story_type, f"共{total}张图片写真集。")


def save_story(album_id: str, data: Dict) -> None:
    """Save story data to local file."""
    story_dir = Path(__file__).parent
    story_dir.mkdir(parents=True, exist_ok=True)

    output_file = story_dir / f"{album_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Story Detection")
    parser.add_argument("--album", type=str, help="Album name")
    parser.add_argument("--all", action="store_true", help="Process all albums")
    parser.add_argument("-d", "--downloads", default="../downloads", help="Downloads directory")
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES, help="Minimum frames for story detection")

    args = parser.parse_args()

    downloads_dir = Path(args.downloads)

    if args.album:
        album_dir = downloads_dir / args.album
        if not album_dir.exists():
            print(f"Album not found: {args.album}")
            return

        result = analyze_album(album_dir, args.min_frames)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        save_story(args.album, result)

    elif args.all:
        for album_dir in sorted(downloads_dir.iterdir()):
            if not album_dir.is_dir():
                continue

            result = analyze_album(album_dir, args.min_frames)
            save_story(album_dir.name, result)


if __name__ == "__main__":
    main()