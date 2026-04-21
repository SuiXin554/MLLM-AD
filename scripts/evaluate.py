from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import cls_metrics

GRID_WORDS = ["左上", "上中", "右上", "左中", "中部", "右中", "左下", "下中", "右下"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_config.yaml")
    return p.parse_args()


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_eval_samples(rows: list[dict]) -> list[dict]:
    by_id = {}
    for r in rows:
        sid = r["sample_id"]
        if sid not in by_id:
            by_id[sid] = {
                "sample_id": sid,
                "image": r["image"],
                "is_anomaly": "异常" if r["is_anomaly"] else "正常",
                "defect_type": r["defect_type"],
                "location_grid": r["location_grid"],
            }
    return list(by_id.values())


def ask(processor, model, image_path: str, question: str, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = outputs[:, inputs.input_ids.shape[1] :]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return text.strip()


def parse_abnormal(text: str) -> str:
    if "正常" in text and "异常" not in text:
        return "正常"
    if "异常" in text:
        return "异常"
    return "异常"


def parse_defect_type(text: str, gt: str) -> str:
    if gt != "无" and gt in text:
        return gt
    if "正常" in text or "无" in text:
        return "无"
    return text[:20]


def parse_location(text: str) -> str:
    for w in GRID_WORDS:
        if w in text:
            return w
    return "中部"


def save_json(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    random.seed(int(cfg.get("seed", 42)))

    rows = read_jsonl(cfg["data"]["val_jsonl"])
    samples = build_eval_samples(rows)
    max_samples = cfg["data"].get("max_samples")
    if isinstance(max_samples, int) and max_samples > 0:
        random.shuffle(samples)
        samples = samples[:max_samples]

    base_model = cfg["model"]["base_model"]
    local_files_only = bool(cfg["model"].get("local_files_only", True))
    max_new_tokens = int(cfg["model"].get("max_new_tokens", 64))

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, local_files_only=local_files_only)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    adapter = cfg["model"].get("lora_adapter")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)

    gt_abn, pr_abn = [], []
    gt_typ, pr_typ = [], []
    gt_loc, pr_loc = [], []
    cases = []

    for s in samples:
        a = ask(processor, model, s["image"], "图中是否存在异常？仅回答正常或异常。", max_new_tokens)
        t = ask(processor, model, s["image"], "请判断该图像的缺陷类型。", max_new_tokens)
        l = ask(processor, model, s["image"], "缺陷主要位于图像哪个区域？", max_new_tokens)

        pa, pt, pl = parse_abnormal(a), parse_defect_type(t, s["defect_type"]), parse_location(l)
        gt_abn.append(s["is_anomaly"])
        pr_abn.append(pa)
        gt_typ.append(s["defect_type"])
        pr_typ.append(pt)
        gt_loc.append(s["location_grid"])
        pr_loc.append(pl)

        cases.append(
            {
                "sample_id": s["sample_id"],
                "image": s["image"],
                "gt": {"abnormal": s["is_anomaly"], "defect_type": s["defect_type"], "location": s["location_grid"]},
                "pred": {"abnormal": pa, "defect_type": pt, "location": pl},
                "raw": {"abnormal": a, "defect_type": t, "location": l},
            }
        )

    m_abn = cls_metrics(gt_abn, pr_abn)
    m_typ = cls_metrics(gt_typ, pr_typ)
    loc_acc = sum(int(g == p) for g, p in zip(gt_loc, pr_loc)) / max(1, len(gt_loc))

    metrics = {
        "num_samples": len(samples),
        "abnormal": {"accuracy": m_abn.accuracy, "f1_macro": m_abn.f1_macro},
        "defect_type": {"accuracy": m_typ.accuracy, "f1_macro": m_typ.f1_macro},
        "location": {"accuracy": loc_acc},
    }

    save_json(cfg["output"]["metrics_json"], metrics)
    save_jsonl(cfg["output"]["cases_jsonl"], cases)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"metrics_json: {cfg['output']['metrics_json']}")
    print(f"cases_jsonl: {cfg['output']['cases_jsonl']}")


if __name__ == "__main__":
    main()
