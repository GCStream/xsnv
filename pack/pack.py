import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterator
import gc

import pyarrow.parquet as pq
from datasets import Dataset, Image


def get_model_name(metadata: dict) -> str:
    return metadata.get("model_name", "")


def scan_albums(downloads_dir: Path, labels_dir: Path) -> Iterator[dict]:
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    labels_by_album = {}

    if labels_dir.exists():
        for label_file in labels_dir.iterdir():
            if label_file.suffix == ".json" and "_labels" in label_file.name:
                album_id = label_file.stem.split("_")[0]
                try:
                    with open(label_file, "r", encoding="utf-8") as f:
                        labels_by_album[album_id] = json.load(f)
                except Exception:
                    pass

    for album_path in sorted(downloads_dir.iterdir()):
        if not album_path.is_dir():
            continue
        metadata_file = album_path / "metadata.json"
        if not metadata_file.exists():
            continue

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        images = [
            f for f in album_path.iterdir() if f.suffix.lower() in image_extensions
        ]
        if not images:
            continue

        model_name = get_model_name(metadata)
        album_id = metadata.get("album_id", album_path.name.split("_")[0])
        label_data = labels_by_album.get(album_id, {})

        for img in images:
            yield {
                "image": str(img.absolute()),
                "file_name": img.name,
                "title": metadata.get("title", ""),
                "model_name": model_name,
                "tags": metadata.get("tags", []),
                "album_id": album_id,
                "text_en": "",
                "text_cn": "",
                "album_score": label_data.get("albumScore"),
                "album_reason": label_data.get("albumNotes", ""),
                "ai_score": label_data.get("aiScore"),
                "ai_reason": label_data.get("aiReason", ""),
                "has_face": label_data.get("has_face"),
                "has_fullbody": label_data.get("has_fullbody"),
            }


def write_shard(records: list, shard_idx: int, temp_dir: Path):
    if not records:
        return

    data_structure = {k: [r[k] for r in records] for k in records[0].keys()}

    ds = Dataset.from_dict(data_structure)
    ds = ds.cast_column("image", Image())

    filepath = temp_dir / f"dataset-{shard_idx:04d}.parquet.tmp"
    ds.to_parquet(str(filepath))

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  shard-{shard_idx:04d}: {len(records)} rows, {size_mb:.1f} MB")

    del ds
    gc.collect()


def pack_dataset(
    downloads_dir: str = "../downloads",
    labels_dir: str = "../labels",
    output_dir: str = "./dataset",
    target_rg_size_mb: int = 200,
    smoke: bool = False,
) -> None:
    downloads_path = Path(downloads_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    print(f"Scanning albums...")
    all_records = list(scan_albums(downloads_path, labels_path))
    if not all_records:
        return

    random.shuffle(all_records)
    total_images = len(all_records)

    sample_size = min(50, total_images)
    avg_size = (
        sum(Path(r["image"]).stat().st_size for r in all_records[:sample_size])
        / sample_size
    )
    target_rows_per_shard = int((target_rg_size_mb * 1024 * 1024) / (avg_size + 2000))

    print(
        f"Total: {total_images} | Avg Size: {avg_size / 1024:.1f}KB | Target Shard: ~{target_rows_per_shard} rows"
    )

    temp_dir = output_path.parent / f"{output_path.name}_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, total_images, target_rows_per_shard):
        shard_idx = i // target_rows_per_shard
        chunk = all_records[i : i + target_rows_per_shard]
        write_shard(chunk, shard_idx, temp_dir)
        if smoke:
            break

    temp_files = sorted(temp_dir.glob("*.tmp"))
    total_shards = len(temp_files)
    for idx, temp_file in enumerate(temp_files):
        final_name = f"train-{idx:05d}-of-{total_shards:05d}.parquet"
        shutil.move(str(temp_file), str(output_path / final_name))

    shutil.rmtree(temp_dir)

    with open(output_path / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            clean_meta = {k: v for k, v in r.items() if k != "image"}
            clean_meta["file_name"] = r["file_name"]
            f.write(json.dumps(clean_meta, ensure_ascii=False) + "\n")

    print(f"\nDone! Files saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pack images into HuggingFace dataset")
    parser.add_argument(
        "--downloads", "-d", default="../downloads", help="Downloads directory"
    )
    parser.add_argument("--labels", "-l", default="../labels", help="Labels directory")
    parser.add_argument("--output", "-o", default="./dataset", help="Output directory")
    parser.add_argument(
        "--target-rg-size", type=int, default=200, help="Target row group size in MB"
    )
    parser.add_argument(
        "--smoke", "-s", action="store_true", help="Only build first shard for testing"
    )

    args = parser.parse_args()

    pack_dataset(
        downloads_dir=args.downloads,
        labels_dir=args.labels,
        output_dir=args.output,
        target_rg_size_mb=args.target_rg_size,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
