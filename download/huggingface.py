#!/usr/bin/env python3
"""
Download and unpack HuggingFace dataset to folder structure.

Usage:
    python huggingface.py DownFlow/meizi ../downloads
    python huggingface.py DownFlow/meizi ../downloads --split train --max 100
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Download HuggingFace dataset")
    parser.add_argument("dataset", help="Dataset ID (e.g., DownFlow/meizi)")
    parser.add_argument(
        "output", nargs="?", default="../downloads", help="Output directory"
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--max", type=int, default=None, help="Max images")
    parser.add_argument("--token", default=None, help="HuggingFace token")
    return parser.parse_args()


def download_dataset(
    dataset_id: str,
    output_dir: str,
    split: str = "train",
    max_images: int = None,
    token: str = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if token is None:
        token = os.environ.get("HF_TOKEN")

    print(f"Loading dataset {dataset_id}...")
    ds = load_dataset(dataset_id, split=split, token=token)
    print(f"Dataset has {len(ds)} images")

    albums = {}
    for idx, row in enumerate(ds):
        if max_images and idx >= max_images:
            break

        album_id = row.get("album_id", "unknown")
        if album_id not in albums:
            albums[album_id] = {
                "title": row.get("title", f"album_{album_id}"),
                "model_name": row.get("model_name", ""),
                "tags": row.get("tags", []),
                "album_id": album_id,
                "images": [],
            }

        albums[album_id]["images"].append(
            {
                "file_name": row.get("file_name", f"{idx:03d}.jpg"),
                "image": row["image"],
                "album_score": row.get("album_score"),
                "album_reason": row.get("album_reason", ""),
                "ai_score": row.get("ai_score"),
                "ai_reason": row.get("ai_reason", ""),
                "has_face": row.get("has_face"),
                "has_fullbody": row.get("has_fullbody"),
            }
        )

    # Sort images within each album by filename (numeric) - dataset images are shuffled
    for album_id, album_data in albums.items():
        album_data["images"].sort(key=lambda x: x["file_name"])

    print(f"Found {len(albums)} albums")

    for album_id, album_data in albums.items():
        folder_name = f"{album_id}_{album_data['title'][:50].replace('/', '_')}"
        album_dir = output_path / folder_name
        album_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "album_id": album_id,
            "title": album_data["title"],
            "model_name": album_data["model_name"],
            "tags": album_data["tags"],
            "image_count": len(album_data["images"]),
        }
        with open(album_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        for img_data in album_data["images"]:
            img = img_data["image"]
            filename = img_data["file_name"]
            img.save(album_dir / filename)

            img_label = {}
            if img_data.get("album_score") is not None:
                img_label["album_score"] = img_data["album_score"]
            if img_data.get("album_reason"):
                img_label["album_reason"] = img_data["album_reason"]
            if img_data.get("ai_score") is not None:
                img_label["ai_score"] = img_data["ai_score"]
            if img_data.get("ai_reason"):
                img_label["ai_reason"] = img_data["ai_reason"]
            if img_data.get("has_face") is not None:
                img_label["has_face"] = img_data["has_face"]
            if img_data.get("has_fullbody") is not None:
                img_label["has_fullbody"] = img_data["has_fullbody"]

            if img_label:
                label_file = album_dir / f"{Path(filename).stem}_labels.json"
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(img_label, f, ensure_ascii=False, indent=2)

        print(f"  - {folder_name}: {len(album_data['images'])} images")

    print(f"\nDone! Downloaded to {output_dir}")


def main():
    args = parse_args()
    download_dataset(args.dataset, args.output, args.split, args.max, args.token)


if __name__ == "__main__":
    main()
