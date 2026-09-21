#!/usr/bin/env python3
"""Download CUB-200-2011 and convert it to multimodal JEV JSONL records.

The generated records use four single-token numeric markers (0-3) for the
species decision and keep the bird box as a normalized xyxy regression target.
The official CUB test split is preserved; validation is sampled only from the
official training split.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tarfile
import urllib.request
from pathlib import Path


ARCHIVE_URLS = [
    # Official CaltechDATA record; the mirror is useful when the signed
    # Caltech object-storage redirect is temporarily unavailable.
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
    "https://media.githubusercontent.com/media/vignagajan/CUB-200-2011/main/CUB_200_2011.tgz",
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "cub-jev-data-preparer/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def ensure_extracted(root: Path) -> Path:
    extracted = root / "CUB_200_2011"
    if extracted.exists():
        return extracted
    archive = root / "CUB_200_2011.tgz"
    if not archive.exists():
        last_error = None
        for url in ARCHIVE_URLS:
            try:
                download(url, archive)
                break
            except Exception as error:
                last_error = error
                if archive.exists():
                    archive.unlink()
        else:
            raise RuntimeError("Could not download CUB-200-2011") from last_error
    print(f"Extracting {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        # Keep compatibility with Python 3.9 while rejecting path traversal.
        root_resolved = root.resolve()
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        tar.extractall(root)
    return extracted


def read_pairs(path: Path, count: int = 2) -> dict[int, list[str]]:
    result = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=count - 1)
        result[int(parts[0])] = parts[1:]
    return result


def read_bbox(path: Path) -> dict[int, tuple[float, float, float, float]]:
    result = {}
    for line in path.read_text().splitlines():
        image_id, x, y, width, height = line.split()
        result[int(image_id)] = tuple(map(float, (x, y, width, height)))
    return result


def load_metadata(dataset: Path):
    images = read_pairs(dataset / "images.txt", 2)
    labels = read_pairs(dataset / "image_class_labels.txt", 2)
    split = read_pairs(dataset / "train_test_split.txt", 2)
    bboxes = read_bbox(dataset / "bounding_boxes.txt")
    classes = {
        int(class_id): name_parts[0]
        for class_id, name_parts in read_pairs(dataset / "classes.txt", 2).items()
    }
    records = []
    for image_id, image_name_parts in images.items():
        image_name = image_name_parts[0]
        class_id = int(labels[image_id][0])
        x, y, w, h = bboxes[image_id]
        # CUB images are JPEGs; read dimensions lazily with Pillow below.
        records.append({
            "image_id": image_id,
            "image_name": image_name,
            "class_id": class_id,
            "class_name": classes[class_id],
            "is_train": split[image_id][0] == "1",
            "bbox_xywh": [x, y, w, h],
        })
    return records


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as image:
        return image.size


def slug(name: str) -> str:
    name = re.sub(r"^\d+\.", "", name)
    return name.lower().replace(" ", "_").replace("/", "_")


def make_record(item, image_root: Path, choices: list[str], split: str) -> dict:
    width, height = image_size(image_root / item["image_name"])
    x, y, w, h = item["bbox_xywh"]
    target = slug(item["class_name"])
    answer_index = choices.index(target)
    option_text = " ".join(f"{i}: {name.replace('_', ' ')}" for i, name in enumerate(choices))
    return {
        "id": f"cub_{item['image_id']:05d}",
        "image": str((image_root / item["image_name"]).resolve()),
        "split": split,
        "messages": [{
            "role": "user",
            "content": f"Which bird species is shown? {option_text} Answer with one digit: 0, 1, 2, or 3.",
        }],
        "options": choices,
        "answer_index": answer_index,
        "answer_marker": str(answer_index),
        "box": [
            max(0.0, min(1.0, x / width)),
            max(0.0, min(1.0, y / height)),
            max(0.0, min(1.0, (x + w) / width)),
            max(0.0, min(1.0, (y + h) / height)),
        ],
        "class_name": item["class_name"],
        "image_width": width,
        "image_height": height,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/cub_records"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = ensure_extracted(args.data_root)
    items = load_metadata(dataset)
    rng = random.Random(args.seed)
    class_names = sorted({slug(item["class_name"]) for item in items})
    train_items = [item for item in items if item["is_train"]]
    test_items = [item for item in items if not item["is_train"]]
    rng.shuffle(train_items)
    val_count = max(1, int(len(train_items) * args.val_fraction))
    val_items, train_items = train_items[:val_count], train_items[val_count:]

    def write(items, name):
        path = args.output / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for item in items:
                distractors = [c for c in class_names if c != slug(item["class_name"])]
                choices = [slug(item["class_name"])] + rng.sample(distractors, 3)
                rng.shuffle(choices)
                handle.write(json.dumps(make_record(item, dataset / "images", choices, name)) + "\n")
        print(f"Wrote {len(items)} records to {path}")

    write(train_items, "train")
    write(val_items, "validation")
    write(test_items, "test")
    (args.output / "classes.json").write_text(json.dumps(class_names, indent=2) + "\n")


if __name__ == "__main__":
    main()
