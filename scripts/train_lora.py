from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from datasets import Dataset


def _parse_semver(version_text: str | None) -> tuple[int, int, int]:
    if not version_text:
        return 0, 0, 0
    clean = str(version_text).split("+")[0]
    nums = re.findall(r"\d+", clean)
    major = int(nums[0]) if len(nums) > 0 else 0
    minor = int(nums[1]) if len(nums) > 1 else 0
    patch = int(nums[2]) if len(nums) > 2 else 0
    return major, minor, patch


def _get_package_version(pkg_name: str) -> str | None:
    try:
        v = importlib_metadata.version(pkg_name)
        if v:
            return str(v)
    except Exception:
        pass
    try:
        mod = importlib.import_module(pkg_name.replace("-", "_"))
        v2 = getattr(mod, "__version__", None)
        return str(v2) if v2 else None
    except Exception:
        return None


def check_runtime_compat() -> None:
    torch_v = _parse_semver(torch.__version__)
    tf_v_str = _get_package_version("transformers")
    if not tf_v_str:
        raise RuntimeError("未检测到 transformers，请先安装：pip install transformers==4.50.3")
    tf_v = _parse_semver(tf_v_str)
    hub_v_str = _get_package_version("huggingface-hub")
    if not hub_v_str:
        raise RuntimeError("未检测到 huggingface-hub，请先安装：pip install huggingface-hub==0.36.0")
    hub_v = _parse_semver(hub_v_str)

    if torch_v < (2, 4, 0) and tf_v >= (5, 0, 0):
        raise RuntimeError(
            "当前环境是 torch<2.4 + transformers>=5，transformers 会禁用 PyTorch 后端。\n"
            f"检测到 torch={torch.__version__}, transformers={tf_v_str}。\n"
            "你不需要升级 torch，请执行以下命令降级 transformers 到 4.x：\n"
            "  pip install --upgrade --no-deps 'transformers==4.50.3' 'tokenizers==0.21.1'"
        )
    if tf_v < (5, 0, 0) and not ((0, 24, 0) <= hub_v < (1, 0, 0)):
        raise RuntimeError(
            "检测到 transformers 4.x 与 huggingface-hub 版本不兼容。\n"
            f"当前 transformers={tf_v_str}, huggingface-hub={hub_v_str}。\n"
            "请执行：\n"
            "  pip install --upgrade --no-deps 'huggingface-hub==0.36.0'"
        )
    # Qwen2.5-VL needs newer transformers than many older 4.x builds.
    if tf_v < (4, 50, 0):
        raise RuntimeError(
            "当前 transformers 版本过低，不支持 qwen2_5_vl 架构。\n"
            f"检测到 transformers={tf_v_str}。\n"
            "请执行：\n"
            "  pip install --upgrade --no-deps 'transformers==4.50.3' 'tokenizers==0.21.1'"
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


def build_chat_messages(question: str, answer: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"你是工业质检助手。请根据图像回答问题：{question}"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]


@dataclass
class VLDataCollator:
    processor: Any
    max_seq_length: int
    image_root: str | None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = []
        images = []
        for f in features:
            messages = build_chat_messages(f["question"], f["answer"])
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
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
    local_model_path = cfg["model"].get("local_model_path")
    require_local = bool(cfg["model"].get("require_local_model", False))
    hf_endpoint = cfg["model"].get("hf_endpoint")
    hf_token = cfg["model"].get("hf_token")
    local_files_only = bool(cfg["model"].get("local_files_only", False))

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = str(hf_endpoint)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    model_ref = local_model_path if local_model_path else model_name
    if local_model_path:
        local_dir = Path(local_model_path)
        if not local_dir.exists():
            raise RuntimeError(f"local_model_path 不存在: {local_model_path}")
        local_files_only = True
    elif require_local:
        raise RuntimeError("配置要求 require_local_model=true，但未设置 local_model_path。")
    use_qlora = bool(cfg["model"].get("use_qlora", True))
    strict_qlora = bool(cfg["model"].get("strict_qlora", False))
    dtype_name = str(cfg["model"].get("torch_dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name)

    quant_config = None
    if use_qlora:
        try:
            _ = importlib_metadata.version("bitsandbytes")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        except importlib_metadata.PackageNotFoundError as e:
            if strict_qlora:
                raise RuntimeError(
                    "配置要求 strict_qlora=true，但环境中未安装 bitsandbytes。\n"
                    "请执行：pip install bitsandbytes==0.44.1"
                ) from e
            print(
                "[train_lora] WARNING: bitsandbytes 未安装，自动降级为普通 LoRA（use_qlora=false）。\n"
                "如需 QLoRA，请安装：pip install bitsandbytes==0.44.1"
            )
            use_qlora = False
    print(f"[train_lora] mode={'QLoRA' if use_qlora else 'LoRA'}")

    try:
        processor = AutoProcessor.from_pretrained(
            model_ref,
            trust_remote_code=True,
            token=hf_token,
            local_files_only=local_files_only,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_ref,
            trust_remote_code=True,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map="auto",
            token=hf_token,
            local_files_only=local_files_only,
        )
    except Exception as e:
        if "qwen2_5_vl" in str(e):
            raise RuntimeError(
                "检测到 `qwen2_5_vl` 架构无法识别：当前 transformers 版本过低。\n"
                "请执行：\n"
                "  pip install --upgrade --no-deps 'transformers==4.50.3' 'tokenizers==0.21.1'"
            ) from e
        raise RuntimeError(
            "加载模型失败：无法从 Hugging Face 拉取或本地模型不可用。\n"
            f"model_ref={model_ref}, local_files_only={local_files_only}\n"
            "解决方案：\n"
            "1) 设置 model.local_model_path 为已下载模型目录；或\n"
            "2) 设置 model.hf_endpoint 为可访问镜像（如 https://hf-mirror.com）；或\n"
            "3) 先手动下载模型到本地，再用 local_files_only=true 运行。"
        ) from e

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
    model.enable_input_require_grads()
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
        label_names=["labels"],
        remove_unused_columns=False,
        eval_strategy="steps" if val_ds is not None else "no",
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
