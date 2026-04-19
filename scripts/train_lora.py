from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from datasets import Dataset


def _parse_semver(version_text: str) -> tuple[int, int, int]:
    clean = version_text.split("+")[0]
    nums = clean.split(".")
    major = int(nums[0]) if len(nums) > 0 and nums[0].isdigit() else 0
    minor = int(nums[1]) if len(nums) > 1 and nums[1].isdigit() else 0
    patch = int(nums[2]) if len(nums) > 2 and nums[2].isdigit() else 0
    return major, minor, patch


def check_runtime_compat() -> None:
    torch_v = _parse_semver(torch.__version__)
    tf_v_str = importlib_metadata.version("transformers")
    tf_v = _parse_semver(tf_v_str)
    hub_v_str = importlib_metadata.version("huggingface-hub")
    hub_v = _parse_semver(hub_v_str)

    if torch_v < (2, 4, 0) and tf_v >= (5, 0, 0):
        raise RuntimeError(
            "当前环境是 torch<2.4 + transformers>=5，transformers 会禁用 PyTorch 后端。\n"
            f"检测到 torch={torch.__version__}, transformers={tf_v_str}。\n"
            "你不需要升级 torch，请执行以下命令降级 transformers 到 4.x：\n"
            "  pip install --upgrade --no-deps 'transformers==4.47.1' 'tokenizers==0.21.0'"
        )
    if tf_v < (5, 0, 0) and not ((0, 24, 0) <= hub_v < (1, 0, 0)):
        raise RuntimeError(
            "检测到 transformers 4.x 与 huggingface-hub 版本不兼容。\n"
            f"当前 transformers={tf_v_str}, huggingface-hub={hub_v_str}。\n"
            "请执行：\n"
            "  pip install --upgrade --no-deps 'huggingface-hub==0.36.0'"
        )
    try:
        import safetensors  # noqa: F401
        import safetensors.torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "safetensors 二进制扩展加载失败（常见于 pip/conda 混装后 ABI 不一致）。\n"
            f"原始错误: {e}\n"
            "请执行：\n"
            "  pip install --force-reinstall --no-cache-dir --no-deps 'safetensors==0.4.5'"
        ) from e


check_runtime_compat()

try:
    from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
except Exception as e:  # pragma: no cover
    if "_safe_open_handle" in str(e):
        raise RuntimeError(
            "transformers 导入失败，根因是 safetensors 二进制不匹配。\n"
            "请执行：\n"
            "  pip install --force-reinstall --no-cache-dir --no-deps 'safetensors==0.4.5'"
        ) from e
    raise

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # transformers 4.x compatibility fallback
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Qwen2.5-VL with LoRA/QLoRA on generated JSONL.")
    p.add_argument("--config", type=str, default="configs/train_lora.yaml")
    return p.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def apply_sample_limit(rows: list[dict[str, Any]], max_n: int | None, seed: int) -> list[dict[str, Any]]:
    if max_n is None or max_n <= 0 or len(rows) <= max_n:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    idx = idx[:max_n]
    return [rows[i] for i in idx]


def _join_image_path(image_path: str, image_root: str | None) -> str:
    p = Path(image_path)
    if p.is_absolute() or not image_root:
        return str(p)
    return str((Path(image_root) / p).resolve())


def build_prompt(question: str, answer: str) -> str:
    return (
        "你是工业质检助手。请根据图像回答问题。\n"
        f"问题：{question}\n"
        f"答案：{answer}"
    )


@dataclass
class VLDataCollator:
    processor: Any
    max_seq_length: int
    image_root: str | None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = []
        images = []
        for f in features:
            prompt = build_prompt(f["question"], f["answer"])
            img_path = _join_image_path(f["image"], self.image_root)
            img = Image.open(img_path).convert("RGB")
            texts.append(prompt)
            images.append(img)

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def build_model_and_processor(cfg: dict[str, Any]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_name = cfg["model"]["name_or_path"]
    use_qlora = bool(cfg["model"].get("use_qlora", True))
    dtype_name = str(cfg["model"].get("torch_dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name)

    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto",
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))

    train_rows = load_jsonl(cfg["data"]["train_jsonl"])
    val_rows = load_jsonl(cfg["data"]["val_jsonl"])
    train_rows = apply_sample_limit(train_rows, cfg["data"].get("max_train_samples"), seed)
    val_rows = apply_sample_limit(val_rows, cfg["data"].get("max_val_samples"), seed)

    if len(train_rows) == 0:
        raise ValueError("train_jsonl is empty; please run scripts/prepare_data.py first.")

    model, processor = build_model_and_processor(cfg)
    collator = VLDataCollator(
        processor=processor,
        max_seq_length=int(cfg["train"].get("max_seq_length", 768)),
        image_root=cfg["data"].get("image_root"),
    )

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows) if len(val_rows) > 0 else None

    tcfg = cfg["train"]
    training_args = TrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=float(tcfg["num_train_epochs"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        eval_steps=int(tcfg.get("eval_steps", 100)),
        save_steps=int(tcfg.get("save_steps", 100)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 2)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        bf16=bool(tcfg.get("bf16", True)),
        fp16=bool(tcfg.get("fp16", False)),
        report_to=tcfg.get("report_to", "none"),
        remove_unused_columns=False,
        evaluation_strategy="steps" if val_ds is not None else "no",
        save_strategy="steps",
        logging_strategy="steps",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(tcfg["output_dir"])
    processor.save_pretrained(tcfg["output_dir"])
    print(f"[train_lora] done. model saved to: {tcfg['output_dir']}")


if __name__ == "__main__":
    main()
