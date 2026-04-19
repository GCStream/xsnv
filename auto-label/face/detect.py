#!/usr/bin/env python3
"""
Face Detection using InsightFace

Usage:
    python -m face.detect --album 12345
    python -m face.detect --all
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

try:
    from arcface import ArcFace
except ImportError:
    ArcFace = None


MODEL_NAME = "buffalo_l"
FACE_APP = None


def get_face_app() -> FaceAnalysis:
    global FACE_APP
    if FACE_APP is None:
        FACE_APP = FaceAnalysis(name=MODEL_NAME, providers=['CPU', 'CUDAExecutionProvider'])
        FACE_APP.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_APP


def load_image(image_path: Path) -> Image.Image:
    """Load and verify image."""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        return None


def detect_faces(image_path: Path) -> Dict:
    """Detect faces in a single image."""
    img = load_image(image_path)
    if img is None:
        return {"error": "Failed to load image", "faces": []}

    app = get_face_app()
    faces = app.get(np.array(img))

    results = []
    for face in faces:
        results.append({
            "bbox": face.bbox.tolist(),
            "score": float(face.score),
            "embedding": face.embedding.tolist() if hasattr(face, 'embedding') else None,
            "age": getattr(face, 'age', None),
            "gender": int(getattr(face, 'gender', -1)) if hasattr(face, 'gender') else -1,
        })

    return {
        "image": str(image_path),
        "has_face": len(results) > 0,
        "face_count": len(results),
        "faces": results,
    }


def get_face_embedding(face_result: Dict) -> Optional[np.ndarray]:
    """Extract embedding from face result."""
    if face_result.get("embedding") is None:
        return None
    return np.array(face_result["embedding"])


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def cluster_faces(face_embeddings: List[np.ndarray], threshold: float = 0.5) -> List[List[int]]:
    """Cluster face embeddings into groups."""
    if not face_embeddings:
        return []

    n = len(face_embeddings)
    if n == 1:
        return [[0]]

    # Build similarity matrix
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(face_embeddings[i], face_embeddings[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    # Simple clustering: group by max similarity to any in cluster
    clusters = []
    assigned = set()

    for i in range(n):
        if i in assigned:
            continue

        cluster = [i]
        for j in range(i + 1, n):
            if j in assigned:
                continue
            if sim_matrix[i, j] >= threshold:
                cluster.append(j)
                assigned.add(j)

        clusters.append(cluster)
        assigned.add(i)

    return clusters


def analyze_album(album_dir: Path, noise_dir: Path) -> Dict:
    """Analyze all faces in an album."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([f for f in album_dir.iterdir() if f.suffix.lower() in image_extensions])

    if len(images) < 3:
        return {"error": "Too few images for analysis", "images": []}

    # Detect faces in all images
    face_results = []
    valid_images = []

    for img_path in images:
        result = detect_faces(img_path)
        face_results.append(result)

        if result.get("has_face"):
            valid_images.append({
                "path": str(img_path),
                "result": result,
            })

    if not valid_images:
        return {
            "album": album_dir.name,
            "has_main_face": False,
            "images": [],
        }

    # Extract embeddings for clustering
    embeddings = []
    for vr in valid_images:
        for face in vr["result"]["faces"]:
            if face.get("embedding"):
                embeddings.append({
                    "image_path": vr["path"],
                    "embedding": np.array(face["embedding"]),
                    "score": face["score"],
                })

    if len(embeddings) < 2:
        # Only one face found, assume main character
        return {
            "album": album_dir.name,
            "has_main_face": True,
            "main_person_embedding": embeddings[0]["embedding"].tolist() if embeddings else None,
            "image_count": len(valid_images),
            "images": valid_images,
        }

    # Cluster embeddings
    emb_list = [e["embedding"] for e in embeddings]
    clusters = cluster_faces(emb_list, threshold=0.5)

    # Find main cluster (most faces)
    main_cluster_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
    main_cluster = clusters[main_cluster_idx]

    # Get representative face from main cluster (highest score)
    main_embeddings = [emb_list[i] for i in main_cluster]
    main_emb = main_embeddings[0]  # Simplified: take first

    # Identify outliers (not in main cluster)
    outlier_indices = set(range(len(embeddings))) - set(main_cluster)

    # Move outlier images to noise
    noise_subdir = noise_dir / "non_main_face" / album_dir.name
    noise_subdir.mkdir(parents=True, exist_ok=True)

    moved = []
    for idx in outlier_indices:
        src_path = embeddings[idx]["image_path"]
        src = Path(src_path)
        if src.exists():
            dst = noise_subdir / src.name
            src.rename(dst)
            moved.append(str(dst))

    return {
        "album": album_dir.name,
        "has_main_face": len(main_cluster) > 0,
        "main_person_embedding": main_emb.tolist(),
        "cluster_count": len(clusters),
        "main_cluster_size": len(main_cluster),
        "outlier_count": len(outlier_indices),
        "outliers_moved": moved,
        "image_count": len(valid_images),
        "images": valid_images[:10],  # Sample for JSON
    }


def save_embeddings(album_id: str, data: Dict) -> None:
    """Save face embeddings to local file."""
    embeddings_dir = Path(__file__).parent / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    output_file = embeddings_dir / f"{album_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Face Detection with InsightFace")
    parser.add_argument("--album", type=str, help="Album name")
    parser.add_argument("--all", action="store_true", help="Process all albums")
    parser.add_argument("-d", "--downloads", default="../downloads", help="Downloads directory")
    parser.add_argument("--noise", default="noise", help="Noise directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")

    args = parser.parse_args()

    downloads_dir = Path(args.downloads)
    noise_dir = Path(args.noise)
    noise_dir.mkdir(parents=True, exist_ok=True)

    if args.album:
        album_dir = downloads_dir / args.album
        if not album_dir.exists():
            print(f"Album not found: {args.album}")
            return

        result = analyze_album(album_dir, noise_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        save_embeddings(args.album, result)

    elif args.all:
        for album_dir in sorted(downloads_dir.iterdir()):
            if not album_dir.is_dir():
                continue

            print(f"Processing {album_dir.name}...")
            result = analyze_album(album_dir, noise_dir)
            save_embeddings(album_dir.name, result)
            print(f"  -> main_face: {result.get('has_main_face')}, outliers: {result.get('outlier_count', 0)}")


if __name__ == "__main__":
    main()