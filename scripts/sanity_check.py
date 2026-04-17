from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_KEYS = {
    "sample_id",
    "image",
    "dataset",
    "object_category",
    "is_anomaly",
    "defect_type",
    "location_grid",
    "location_source",
    "task",
    "question",
    "answer",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick sanity check for generated JSONL")
    p.add_argument("--jsonl", type=str, default="data/processed/train.jsonl")
    p.add_argument("--max_check", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    p = Path(args.jsonl)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    n = 0
    missing = 0
    missing_images = 0
    tasks = {}

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            row = json.loads(line)
            ks = set(row.keys())
            if not REQUIRED_KEYS.issubset(ks):
                missing += 1
            img = Path(row["image"])
            if not img.exists():
                missing_images += 1
            task = row.get("task", "unknown")
            tasks[task] = tasks.get(task, 0) + 1
            if n >= args.max_check:
                break

    print(f"[sanity] checked_rows={n}")
    print(f"[sanity] missing_required_keys={missing}")
    print(f"[sanity] missing_images={missing_images}")
    print(f"[sanity] task_distribution={tasks}")


if __name__ == "__main__":
    main()
