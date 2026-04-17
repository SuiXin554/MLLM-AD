from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.dataset_builder import DatasetBuilder, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MLLM-AD instruction JSONL from MVTec AD and VisA roots.")
    p.add_argument("--config", type=str, default="configs/data_config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    builder = DatasetBuilder(cfg)
    rows, stats = builder.build()
    train, val = builder.split(rows)

    train_path = cfg["output"]["train_jsonl"]
    val_path = cfg["output"]["val_jsonl"]
    stats_path = cfg["output"]["stats_json"]

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    stats.update(
        {
            "num_train_records": len(train),
            "num_val_records": len(val),
            "config": args.config,
        }
    )
    write_json(stats_path, stats)

    print("[prepare_data] done")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"train_jsonl: {train_path}")
    print(f"val_jsonl:   {val_path}")
    print(f"stats_json:  {stats_path}")


if __name__ == "__main__":
    main()
