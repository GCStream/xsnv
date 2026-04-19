#!/usr/bin/env python3
"""
Download albums from xsnvshen.com

Usage:
    python download.py --album 45545
    python download.py --model 28036
    python download.py --tag 116
    python download.py --search "鱼子酱"
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Optional

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.xsnvshen.com"
IMG_CDN = "https://img.xsnvshen.com"
MODEL_CACHE_FILE = Path(__file__).parent.parent / "model_cache.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

IMG_SESSION = requests.Session()
IMG_SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Origin": "https://www.xsnvshen.com",
    }
)

MODEL_CACHE = {}


def init_cache():
    global MODEL_CACHE
    MODEL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_CACHE_FILE.exists():
        with open(MODEL_CACHE_FILE, "r", encoding="utf-8") as f:
            MODEL_CACHE = json.load(f)


def save_cache():
    with open(MODEL_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(MODEL_CACHE, f, ensure_ascii=False, indent=2)


def init_session():
    init_cache()
    SESSION.get(BASE_URL, timeout=30)


def get_album_page(album_id: int) -> bytes:
    url = f"{BASE_URL}/album/{album_id}"
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def extract_name_from_title(title: str) -> Optional[str]:
    if not title:
        return None
    cleaned = title.strip()
    match = re.search(r"\]\s*([^\[\]]+?)(?:\s*[-_]\s*|\s+[A-Za-z])", cleaned)
    if match:
        name = match.group(1).strip()
        if name and len(name) > 1:
            return name
    match = re.search(r"^([^\[\]\-_]+?)(?:\s*[-_]\s*|\s+[A-Za-z0-9])", cleaned)
    if match:
        name = match.group(1).strip()
        if name and len(name) > 1:
            return name
    first_word = cleaned.split()[0] if cleaned.split() else ""
    if len(first_word) >= 2:
        return first_word
    return None


def parse_album_info(html: bytes, album_id: int) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    title = title.get_text(strip=True) if title else f"album_{album_id}"

    tags = []
    seen_tags = set()
    tag_links = soup.select(".tags a")
    for tag in tag_links:
        tag_text = tag.get_text(strip=True)
        if tag_text and tag_text not in seen_tags:
            tags.append(tag_text)
            seen_tags.add(tag_text)

    model_name = ""
    girl_id = None

    model_link = soup.select_one(".girl-link")
    if model_link:
        href = model_link.get("href", "")
        match = re.search(r"/girl/(\d+)", href)
        if match:
            girl_id = int(match.group(1))
            model_name = model_link.get_text(strip=True)
    if not model_name:
        model_name = extract_name_from_title(title) or ""

    return {
        "album_id": album_id,
        "title": title,
        "model_name": model_name,
        "tags": tags,
        "girl_id": girl_id,
    }


def resolve_model_name(girl_id: int) -> Optional[str]:
    if str(girl_id) in MODEL_CACHE:
        return MODEL_CACHE[str(girl_id)]
    try:
        url = f"{BASE_URL}/girl/{girl_id}"
        resp = SESSION.get(url, timeout=30)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            title = soup.find("h1")
            if title:
                name = title.get_text(strip=True)
                MODEL_CACHE[str(girl_id)] = name
                save_cache()
                return name
    except Exception:
        pass
    return None


def backfill_metadata(download_dir: str = "./downloads"):
    init_cache()
    download_path = Path(download_dir)
    if not download_path.exists():
        return

    for album_dir in sorted(download_path.iterdir()):
        if not album_dir.is_dir():
            continue
        metadata_file = album_dir / "metadata.json"
        if not metadata_file.exists():
            continue

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata.get("model_name"):
            continue

        girl_id = metadata.get("girl_id")
        if not girl_id:
            continue

        model_name = resolve_model_name(girl_id)
        if model_name:
            enrich_metadata(album_dir, girl_id)
            print(f"  Updated {album_dir.name}: {model_name}")


def enrich_metadata(album_dir: Path, girl_id: int):
    metadata_file = album_dir / "metadata.json"
    if not metadata_file.exists():
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not metadata.get("model_name"):
        model_name = resolve_model_name(girl_id)
        if model_name:
            metadata["model_name"] = model_name
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)


def get_image_urls(album_id: int, gid: int, count: int) -> List[Tuple[int, str]]:
    html = get_album_page(album_id)
    return get_image_urls_from_page(html, album_id)


def get_image_urls_from_page(html: bytes, album_id: int) -> List[Tuple[int, str]]:
    urls = []
    soup = BeautifulSoup(html, "html.parser")
    area = soup.select_one(".photo-area")
    if not area:
        return urls

    for idx, img in enumerate(area.select("img"), 1):
        src = img.get("src") or img.get("data-src") or ""
        if src and "://" in src:
            url = src.split("?", 1)[0]
            urls.append((idx, url))
        if len(urls) >= 100:
            break

    alt_imgs = re.findall(
        rf"https://img\.xsnvshen\.com/[^\"']+", html.decode("utf-8", errors="ignore")
    )
    for url in alt_imgs[:200]:
        if url.endswith((".jpg", ".jpeg", ".png", ".webp")):
            clean_url = url.split("?", 1)[0]
            idx = len(urls) + 1
            urls.append((idx, clean_url))

    return urls


def download_image(album_id: int, idx: int, url: str, save_dir: Path) -> bool:
    try:
        resp = IMG_SESSION.get(url, timeout=60)
        resp.raise_for_status()

        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"
        elif ".webp" in url.lower():
            ext = ".webp"

        filename = f"{idx:03d}{ext}"
        filepath = save_dir / filename

        with open(filepath, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"Error downloading {album_id}/{idx}: {e}")
        return False


def save_metadata(save_dir: Path, album_info: dict):
    metadata_file = save_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(album_info, f, ensure_ascii=False, indent=2)


def check_album_complete(album_dir: Path, expected_count: int) -> Tuple[bool, int]:
    if not album_dir.exists():
        return False, 0

    metadata_file = album_dir / "metadata.json"
    if not metadata_file.exists():
        return False, 0

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        image_count = metadata.get("image_count", 0)
        return image_count >= expected_count, image_count
    except Exception:
        return False, 0


def mark_complete(album_dir: Path):
    metadata_file = album_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata["completed"] = True
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def get_albums_from_page(url: str, max_pages: int = None) -> List[int]:
    album_ids = []
    for page in range(1, max_pages + 1) if max_pages else range(1, 1000):
        try:
            page_url = f"{url}?page={page}"
            resp = SESSION.get(page_url, timeout=30)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.content, "html.parser")
            links = soup.select(".album-item a")
            if not links:
                break
            for link in links:
                href = link.get("href", "")
                match = re.search(r"/album/(\d+)", href)
                if match:
                    album_id = int(match.group(1))
                    if album_id not in album_ids:
                        album_ids.append(album_id)
        except Exception:
            break
    return album_ids


def download_album(
    album_id: int,
    output_dir: str = "./downloads",
    workers: int = 4,
    delay: float = 0.5,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Fetching album {album_id}...")
    html = get_album_page(album_id)
    album_info = parse_album_info(html, album_id)

    image_urls = get_image_urls_from_page(html, album_id)
    if not image_urls:
        print(f"No images found for album {album_id}")
        return

    album_info["image_count"] = len(image_urls)

    folder_name = f"{album_id}_{album_info['title'][:50].replace('/', '_')}"
    album_dir = output_path / folder_name
    album_dir.mkdir(parents=True, exist_ok=True)

    is_complete, count = check_album_complete(album_dir, len(image_urls))
    if is_complete:
        print(f"Album {album_id} already complete ({count} images)")
        return

    print(f"Downloading {len(image_urls)} images to {folder_name}...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_image, album_id, idx, url, album_dir): (idx, url)
            for idx, url in image_urls
        }
        for future in as_completed(futures):
            future.result()

    save_metadata(album_dir, album_info)

    is_complete, count = check_album_complete(album_dir, len(image_urls))
    if is_complete:
        mark_complete(album_dir)
        print(f"Album {album_id} complete!")

    time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Download albums from xsnvshen.com")
    parser.add_argument("--album", type=int, help="Album ID")
    parser.add_argument("--model", type=int, help="Model ID")
    parser.add_argument("--tag", type=int, help="Tag ID")
    parser.add_argument("--search", type=str, help="Search keyword")
    parser.add_argument(
        "-w", "--workers", type=int, default=4, help="Concurrent workers"
    )
    parser.add_argument(
        "-d", "--delay", type=float, default=0.5, help="Delay between albums"
    )
    parser.add_argument("-p", "--pages", type=int, help="Max pages")
    parser.add_argument(
        "-o", "--output", default="../downloads", help="Output directory"
    )

    args = parser.parse_args()

    init_session()

    if args.album:
        download_album(args.album, args.output, args.workers, args.delay)
    elif args.model:
        url = f"{BASE_URL}/girl/{args.model}/albums"
        album_ids = get_albums_from_page(url, args.pages)
        print(f"Found {len(album_ids)} albums for model {args.model}")
        for album_id in album_ids:
            download_album(album_id, args.output, args.workers, args.delay)
    elif args.tag:
        url = f"{BASE_URL}/tag/{args.tag}"
        album_ids = get_albums_from_page(url, args.pages)
        print(f"Found {len(album_ids)} albums for tag {args.tag}")
        for album_id in album_ids:
            download_album(album_id, args.output, args.workers, args.delay)
    elif args.search:
        url = f"{BASE_URL}/search/{args.search}"
        album_ids = get_albums_from_page(url, args.pages or 5)
        print(f"Found {len(album_ids)} albums for search '{args.search}'")
        for album_id in album_ids:
            download_album(album_id, args.output, args.workers, args.delay)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
