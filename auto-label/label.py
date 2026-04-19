#!/usr/bin/env python3
"""
AI Labeler - Production script for NVIDIA GH200
Full pipeline: Face dedup → Story detect → AI score → Tags

Usage:
    python label.py --album 45545
    python label.py --all
    python label.py --unlabeled
    python label.py --pipeline  # Full pipeline
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

import yaml
import requests
from PIL import Image
import numpy as np

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "huihui-ai/Huihui-Qwen3.5-2B-abliterated")

DOWNLOADS_DIR = os.getenv("DOWNLOADS_DIR", "../downloads")
LABELS_DIR = os.getenv("LABELS_DIR", "../labels")
NOISE_DIR = os.getenv("NOISE_DIR", "noise")
CONFIG_FILE = os.getenv("CONFIG_FILE", "config/scoring_rules.yaml")


def load_config() -> Dict:
    """Load scoring config from YAML."""
    config_path = Path(__file__).parent / CONFIG_FILE
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return get_default_config()


def get_default_config() -> Dict:
    """Default config if YAML not found."""
    return {
        "scoring": {
            "factors": {
                "face": {"weight": 0.30},
                "body": {"weight": 0.15},
                "pose": {"weight": 0.15},
                "clothing": {"weight": 0.10},
                "background": {"weight": 0.10},
                "lighting": {"weight": 0.10},
                "composition": {"weight": 0.10},
            },
            "story_modifier": {
                "key_frame_bonus": 0.2,
                "filler_penalty": -0.1,
            },
            "score_mapping": {
                1: "极差",
                2: "差",
                3: "较差",
                4: "一般-",
                5: "一般",
                6: "一般+",
                7: "较好",
                8: "好",
                9: "极好",
            },
        },
        "tag_categories": {
            "quality": ["高质量", "一般", "模糊"],
            "pose": ["全身", "半身", "特写"],
            "outfit": ["泳装", "内衣", "制服"],
        },
    }


def call_vllm(prompt: str, image: Image.Image = None) -> Dict:
    """Call vLLM API for inference."""
    messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def check_vllm_ready() -> bool:
    """Check if vLLM is running."""
    try:
        resp = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# =============================================================================
# Face Detection Pipeline
# =============================================================================

def run_face_detection(album_dir: Path, noise_dir: Path) -> Dict:
    """Run face detection on album."""
    # Lazy import to avoid loading if not needed
    from face.detect import analyze_album as face_analyze

    try:
        result = face_analyze(album_dir, noise_dir)
        return result
    except Exception as e:
        print(f"  Face detection error: {e}")
        return {"error": str(e)}


# =============================================================================
# Story Detection Pipeline
# =============================================================================

def run_story_detection(album_dir: Path) -> Dict:
    """Run story detection on album."""
    from story.detect import analyze_album as story_analyze

    try:
        result = story_analyze(album_dir)
        return result
    except Exception as e:
        print(f"  Story detection error: {e}")
        return {"error": str(e)}


# =============================================================================
# Noise Filtering
# =============================================================================

def detect_broken_images(album_dir: Path, noise_subdir: Path) -> List[str]:
    """Detect and move broken images."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    broken = []

    noise_subdir.mkdir(parents=True, exist_ok=True)

    for img_path in album_dir.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in image_extensions:
            continue

        try:
            img = Image.open(img_path)
            img.verify()
        except Exception:
            dst = noise_subdir / img_path.name
            shutil.move(str(img_path), str(dst))
            broken.append(str(dst))

    return broken


def detect_watermarks(album_dir: Path, noise_subdir: Path) -> List[str]:
    """Detect images with huge watermarks (placeholder)."""
    # This would need actual watermark detection
    # For now, return empty
    return []


# =============================================================================
# AI Scoring
# =============================================================================

def score_image(image: Image.Image, config: Dict, story_data: Dict = None) -> Dict:
    """Score a single image using vLLM with YAML config."""
    story_modifier = config.get("scoring", {}).get("story_modifier", {})
    factors = config.get("scoring", {}).get("factors", {})

    # Build prompt from config
    factor_str = "\n".join([f"- {k}: weight={v.get('weight', 0)}" for k, v in factors.items()])

    prompt = f"""You are a professional photo critic. Score this image 1-9 based on:

{factor_str}

Consider: {"This is a key frame" if story_data and story_data.get('is_key_frame') else "This is a filler frame"}
{"Key frame bonus: +" + str(story_modifier.get('key_frame_bonus', 0)) if story_data and story_data.get('is_key_frame') else ""}
{"Filler penalty: " + str(story_modifier.get('filler_penalty', 0)) if story_data and not story_data.get('is_key_frame') else ""}

Output format (just these lines):
ai_score: <1-9>
ai_factors: face=<0-1> body=<0-1> pose=<0-1> clothing=<0-1> background=<0-1> lighting=<0-1> composition=<0-1>"""

    result = call_vllm(prompt, image)

    if "error" in result:
        return {"ai_score": 5, "ai_reason": result["error"], "ai_factors": {}}

    try:
        content = result["choices"][0]["message"]["content"]
        lines = content.split("\n")
        score = 5
        factors_out = {}
        reason = ""

        for line in lines:
            line = line.strip()
            if line.startswith("ai_score:"):
                try:
                    score = int(line.split(":")[1].strip())
                    score = max(1, min(9, score))
                except:
                    pass
            elif line.startswith("ai_factors:"):
                parts = line.replace("ai_factors:", "").strip().split()
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=")
                        factors_out[k.strip()] = float(v.strip())
            elif line.startswith("ai_reason:") or line.startswith("reason:"):
                reason = line.split(":", 1)[1].strip()

        return {"ai_score": score, "ai_reason": reason, "ai_factors": factors_out}
    except Exception:
        return {"ai_score": 5, "ai_reason": "Parse error", "ai_factors": {}}


def tag_image(image: Image.Image, config: Dict) -> Dict:
    """Generate tags for image."""
    tag_categories = config.get("tag_categories", {})

    all_tags = []
    for tags in tag_categories.values():
        all_tags.extend(tags)

    tags_str = ", ".join(all_tags[:32])

    prompt = f"""Select the most suitable tags from this list (comma-separated, max 5):

{tags_str}

Only output the tags, nothing else."""

    result = call_vllm(prompt, image)

    if "error" in result:
        return {"ai_tags": []}

    try:
        content = result["choices"][0]["message"]["content"]
        tags = [t.strip() for t in content.split(",") if t.strip()]
        return {"ai_tags": tags[:5]}
    except Exception:
        return {"ai_tags": []}


# =============================================================================
# Main Processing
# =============================================================================

def process_album(
    album_name: str,
    downloads_dir: Path,
    labels_dir: Path,
    noise_dir: Path,
    config: Dict,
    run_face: bool = True,
    run_story: bool = True,
    run_scoring: bool = True,
) -> Dict:
    """Process a single album through the pipeline."""
    album_dir = downloads_dir / album_name

    if not album_dir.exists():
        return {"error": f"Album not found: {album_name}"}

    results = {
        "album": album_name,
        "face_detection": None,
        "story_detection": None,
        "images": {},
    }

    labels_dir.mkdir(parents=True, exist_ok=True)
    label_file = labels_dir / f"{album_name}_labels.json"

    existing_labels = {}
    if label_file.exists():
        with open(label_file, "r", encoding="utf-8") as f:
            existing_labels = json.load(f)

    print(f"Processing {album_name}...")

    # Stage 1: Face detection
    if run_face:
        print(f"  [1/3] Face detection...")
        results["face_detection"] = run_face_detection(album_dir, noise_dir)

    # Stage 2: Story detection
    if run_story:
        print(f"  [2/3] Story detection...")
        story_data = run_story_detection(album_dir)
        results["story_detection"] = story_data

        # Update text_en/text_cn
        if story_data.get("text_en"):
            existing_labels["text_en"] = story_data["text_en"]
        if story_data.get("text_cn"):
            existing_labels["text_cn"] = story_data["text_cn"]

    # Stage 3: Noise filtering
    print(f"  [2.5/3] Noise filtering...")
    broken_dir = noise_dir / "broken" / album_name
    broken = detect_broken_images(album_dir, broken_dir)
    if broken:
        print(f"    Moved {len(broken)} broken images")

    watermark_dir = noise_dir / "watermark" / album_name
    watermarks = detect_watermarks(album_dir, watermark_dir)
    if watermarks:
        print(f"    Moved {len(watermarks)} watermark images")

    # Stage 4: AI Scoring
    if run_scoring:
        print(f"  [3/3] AI scoring...")
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        images = sorted([f for f in album_dir.iterdir() if f.suffix.lower() in image_extensions])

        story_data = results.get("story_detection", {})
        key_frames = story_data.get("key_frames", []) if story_data else []
        is_filler = story_data.get("is_filler", []) if story_data else []

        for img_path in images:
            img_name = img_path.stem

            if existing_labels.get("images", {}).get(img_name, {}).get("ai_score"):
                print(f"    {img_name}: already scored")
                continue

            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Get story context for this image
                idx = images.index(img_path)
                img_story_context = {
                    "is_key_frame": idx in key_frames if key_frames else False,
                    "is_filler": is_filler[idx] if is_filler and idx < len(is_filler) else False,
                }

                # Score
                score_result = score_image(img, config, img_story_context)

                # Tags
                tag_result = tag_image(img, config)

                img_result = {**score_result, **tag_result}

                if "images" not in existing_labels:
                    existing_labels["images"] = {}
                existing_labels["images"][img_name] = img_result

                print(f"    {img_name}: score={score_result.get('ai_score')}, tags={tag_result.get('ai_tags', [])[:2]}")

            except Exception as e:
                print(f"    Error processing {img_name}: {e}")

        # Save labels
        with open(label_file, "w", encoding="utf-8") as f:
            json.dump(existing_labels, f, ensure_ascii=False, indent=2)

    print(f"  Done: {album_name}")
    return results


def main():
    parser = argparse.ArgumentParser(description="AI Labeler - Full Pipeline")
    parser.add_argument("--album", type=str, help="Album name")
    parser.add_argument("--all", action="store_true", help="Process all albums")
    parser.add_argument("--unlabeled", action="store_true", help="Process unlabeled albums")
    parser.add_argument("--pipeline", action="store_true", help="Full pipeline (face + story + scoring)")
    parser.add_argument("--no-face", action="store_true", help="Skip face detection")
    parser.add_argument("--no-story", action="store_true", help="Skip story detection")
    parser.add_argument("--no-scoring", action="store_true", help="Skip AI scoring")
    parser.add_argument("-d", "--downloads", default=DOWNLOADS_DIR, help="Downloads directory")
    parser.add_argument("-l", "--labels", default=LABELS_DIR, help="Labels directory")
    parser.add_argument("-n", "--noise", default=NOISE_DIR, help="Noise directory")

    args = parser.parse_args()

    downloads_dir = Path(args.downloads)
    labels_dir = Path(args.labels)
    noise_dir = Path(args.noise)

    config = load_config()

    print(f"Config loaded: {list(config.get('scoring', {}).get('factors', {}).keys())}")

    if args.pipeline:
        args.no_face = args.no_face or False
        args.no_story = args.no_story or False
        args.no_scoring = args.no_scoring or False
    else:
        args.no_scoring = True  # Default: just process what explicitly asked

    if args.album:
        process_album(
            args.album,
            downloads_dir,
            labels_dir,
            noise_dir,
            config,
            run_face=not args.no_face,
            run_story=not args.no_story,
            run_scoring=not args.no_scoring,
        )
    elif args.all or args.unlabeled:
        for album_dir in sorted(downloads_dir.iterdir()):
            if not album_dir.is_dir():
                continue
            process_album(
                album_dir.name,
                downloads_dir,
                labels_dir,
                noise_dir,
                config,
                run_face=not args.no_face,
                run_story=not args.no_story,
                run_scoring=not args.no_scoring,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()