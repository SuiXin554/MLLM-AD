from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from src.prompt_templates import (
    abnormal_answer,
    defect_type_answer,
    location_answer,
    rationale_answer,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GOOD_KEYS = {"good", "normal", "ok"}
ANOMALY_KEYS = {"anomaly", "defect", "abnormal", "bad", "ng", "crack", "scratch", "fault", "broken"}
GRID = [
    ["左上", "上中", "右上"],
    ["左中", "中部", "右中"],
    ["左下", "下中", "右下"],
]


@dataclass
class BaseSample:
    sample_id: str
    image: str
    dataset: str
    object_category: str
    is_anomaly: bool
    defect_type: str
    location_grid: str
    location_source: str


class DatasetBuilder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.q = cfg["question_templates"]

    def build(self) -> tuple[list[dict], dict]:
        base = []
        mvtec_root = Path(self.cfg["paths"]["mvtec_root"])
        visa_root = Path(self.cfg["paths"]["visa_root"])

        if mvtec_root.exists():
            base.extend(self._scan_mvtec(mvtec_root))
        if visa_root.exists():
            base.extend(self._scan_visa(visa_root))

        max_n = self.cfg.get("max_samples_per_dataset")
        if isinstance(max_n, int) and max_n > 0:
            random.shuffle(base)
            base = base[:max_n]

        random.seed(self.cfg.get("seed", 42))
        random.shuffle(base)

        records = []
        for b in base:
            records.extend(self._to_multitask_records(b))

        stats = self._calc_stats(base, records)
        return records, stats

    def split(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        random.seed(self.cfg.get("seed", 42))
        by_sample = {}
        for r in records:
            by_sample.setdefault(r["sample_id"], []).append(r)

        sample_ids = list(by_sample.keys())
        random.shuffle(sample_ids)
        val_ratio = float(self.cfg.get("val_ratio", 0.1))
        n_val = max(1, int(len(sample_ids) * val_ratio)) if sample_ids else 0
        val_ids = set(sample_ids[:n_val])

        train, val = [], []
        for sid, rs in by_sample.items():
            (val if sid in val_ids else train).extend(rs)
        return train, val

    def _scan_mvtec(self, root: Path) -> list[BaseSample]:
        out: list[BaseSample] = []
        for category in sorted(p for p in root.iterdir() if p.is_dir()):
            category_name = category.name
            for split_name in ["train", "test"]:
                split_dir = category / split_name
                if not split_dir.exists():
                    continue
                for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                    defect_type = defect_dir.name
                    is_anomaly = defect_type != "good"
                    for img in defect_dir.rglob("*"):
                        if img.suffix.lower() not in IMAGE_EXTS:
                            continue
                        mask = self._find_mvtec_mask(category, defect_type, img)
                        loc, source = self._infer_location(img, mask)
                        sid = f"mvtec::{category_name}::{split_name}::{defect_type}::{img.stem}"
                        out.append(
                            BaseSample(
                                sample_id=sid,
                                image=str(img),
                                dataset="mvtec",
                                object_category=category_name,
                                is_anomaly=is_anomaly,
                                defect_type="无" if not is_anomaly else defect_type,
                                location_grid=loc,
                                location_source=source,
                            )
                        )
        return out

    def _find_mvtec_mask(self, category_dir: Path, defect_type: str, img_path: Path) -> Path | None:
        if defect_type == "good":
            return None
        gt_dir = category_dir / "ground_truth" / defect_type
        if not gt_dir.exists():
            return None
        candidates = [
            gt_dir / f"{img_path.stem}_mask.png",
            gt_dir / f"{img_path.stem}.png",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _scan_visa(self, root: Path) -> list[BaseSample]:
        out: list[BaseSample] = []
        for img in root.rglob("*"):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            parts = {p.lower() for p in img.parts}
            if "mask" in img.name.lower() or "ground_truth" in parts or "masks" in parts:
                continue

            is_anomaly = self._infer_is_anomaly(img)
            category = self._infer_category(root, img)
            defect_type = self._infer_defect_type(img, is_anomaly)
            mask = self._find_mask_nearby(img)
            loc, source = self._infer_location(img, mask)
            sid = f"visa::{category}::{defect_type}::{img.stem}"
            out.append(
                BaseSample(
                    sample_id=sid,
                    image=str(img),
                    dataset="visa",
                    object_category=category,
                    is_anomaly=is_anomaly,
                    defect_type="无" if not is_anomaly else defect_type,
                    location_grid=loc,
                    location_source=source,
                )
            )
        return out

    def _infer_is_anomaly(self, img: Path) -> bool:
        lower_parts = [p.lower() for p in img.parts]
        if any(k in part for part in lower_parts for k in GOOD_KEYS):
            return False
        return any(k in part for part in lower_parts for k in ANOMALY_KEYS)

    def _infer_category(self, root: Path, img: Path) -> str:
        rel = img.relative_to(root)
        return rel.parts[0] if rel.parts else "unknown"

    def _infer_defect_type(self, img: Path, is_anomaly: bool) -> str:
        if not is_anomaly:
            return "无"
        skip = {"images", "image", "test", "train", "anomaly", "abnormal", "defect", "bad", "ng"}
        for p in reversed(img.parts[:-1]):
            low = p.lower()
            if low not in skip and low not in GOOD_KEYS:
                return p
        return "unknown_defect"

    def _find_mask_nearby(self, img: Path) -> Path | None:
        parent = img.parent
        stem = img.stem
        candidates = [
            parent / f"{stem}_mask.png",
            parent / f"{stem}.png",
            parent.parent / "masks" / f"{stem}.png",
            parent.parent / "ground_truth" / f"{stem}_mask.png",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _infer_location(self, img_path: Path, mask_path: Path | None) -> tuple[str, str]:
        if mask_path and mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                cx = float(xs.mean())
                cy = float(ys.mean())
                h, w = mask.shape
                return self._xy_to_grid(cx, cy, w, h), "mask"

        with Image.open(img_path) as img:
            w, h = img.size
        return self._xy_to_grid(w / 2, h / 2, w, h), "weak_center"

    def _xy_to_grid(self, x: float, y: float, width: int, height: int) -> str:
        c = min(2, max(0, int(3 * x / max(1, width))))
        r = min(2, max(0, int(3 * y / max(1, height))))
        return GRID[r][c]

    def _to_multitask_records(self, b: BaseSample) -> list[dict]:
        item = asdict(b)
        weak = b.location_source != "mask"
        rows = []

        for q in self.q["abnormal"]:
            rows.append(self._row(item, "abnormal", q, abnormal_answer(b.is_anomaly)))
        for q in self.q["defect_type"]:
            rows.append(self._row(item, "defect_type", q, defect_type_answer(b.is_anomaly, b.defect_type)))
        for q in self.q["location"]:
            rows.append(self._row(item, "location", q, location_answer(b.is_anomaly, b.location_grid, weak)))
        for q in self.q["rationale"]:
            rows.append(self._row(item, "rationale", q, rationale_answer(b.is_anomaly, b.defect_type, b.location_grid)))

        return rows

    def _row(self, item: dict, task: str, question: str, answer: str) -> dict:
        return {
            **item,
            "task": task,
            "question": question,
            "answer": answer,
        }

    @staticmethod
    def _calc_stats(base: list[BaseSample], records: list[dict]) -> dict:
        by_dataset = {}
        for b in base:
            by_dataset.setdefault(b.dataset, 0)
            by_dataset[b.dataset] += 1
        return {
            "num_base_samples": len(base),
            "num_instruction_records": len(records),
            "by_dataset": by_dataset,
        }


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
